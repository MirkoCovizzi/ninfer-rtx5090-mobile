from __future__ import annotations

import struct
from dataclasses import replace

import pytest
import torch

from tools.artifact.reader import Artifact
from tools.artifact.codecs.row_split import decode_row_split_codes
from tools.artifact.codecs.nvfp4 import encode_nvfp4
from tools.artifact.schema import binding_parts
from tools.artifact.writer import ArtifactWriter
from tools.convert.methods import grouped_absmax, import_encoded
from tools.convert.model import Model, Parameter
from tools.artifact.tensor_output import TensorOutput
from tools.convert.recipe import Recipe
from tools.convert.sources.logical import EncodedRows, LogicalSource, array_source
from tools.convert.sources.compressed_tensors import matrix_source
from tools.convert.sources.safetensors import SafetensorsSource
from tools.convert.official_recipes import qwen3_8_27b_quasar_nvfp4


def _model(names=("query", "key", "gate", "value")):
    model = Model({"text": {"config": {}}})
    for index, name in enumerate(names):
        values = (
            ((index + 1) * torch.tensor([1, 2, 4, 8], dtype=torch.bfloat16))[:, None]
            .expand(4, 128)
            .contiguous()
        )
        model.add(
            Parameter(name, (4, 128), array_source(values, name), inputs=("input",))
        )
    model.packing_groups = [
        group for group in (tuple(names), tuple(names[:2]), tuple(names[2:])) if group
    ]
    return model


def _write(path, model, prepared):
    specs = [job.spec for job in prepared.weights] + [
        spec for spec, _ in prepared.auxiliaries
    ]
    with ArtifactWriter(
        path,
        specs,
        components=model.components,
        bindings=prepared.bindings,
        uses=prepared.uses,
    ) as writer:
        for job in prepared.weights:
            job.prepared.produce(TensorOutput(writer, job.spec.id))
        for spec, data in prepared.auxiliaries:
            writer.write_object(spec.id, data)


def _assert_quantized_rows(artifact, part, amplitudes, format):
    object_id, begin, end = part
    obj = artifact.object(object_id)
    assert obj.format == format and end - begin == len(amplitudes) * 128
    scales, codes = decode_row_split_codes(
        artifact.read_object(object_id), format, obj.shape
    )
    first, last = begin // 128, end // 128
    maximum = {"q4_g64_fp16": 7, "q5_g64_fp16": 15}[format]
    assert bool((codes[first:last] == maximum).all())
    expected = [
        struct.unpack("<H", struct.pack("<e", value / maximum))[0]
        for value in amplitudes
    ]
    assert scales[first:last].view(torch.int16).tolist() == [
        [word, word] for word in expected
    ]


def test_mixed_attention_uses_two_parents_and_distinct_permissions(tmp_path):
    model = _model()
    recipe = Recipe(model)
    recipe.assign(
        ("query", "key"),
        format="q4_g64_fp16",
        method=grouped_absmax,
        activation_policy="AllowA4",
    )
    recipe.assign(
        ("gate", "value"),
        format="q5_g64_fp16",
        method=grouped_absmax,
        activation_policy="AllowA8",
    )
    prepared = recipe.prepare(device="cpu", rows_per_chunk=3)
    path = tmp_path / "mixed.ninfer"
    _write(path, model, prepared)
    with Artifact(path) as artifact:
        parts = {
            name: binding_parts(binding, artifact.by_id)[0]
            for name, binding in artifact.directory.bindings.items()
        }
        assert parts["query"][0] == parts["key"][0]
        assert parts["gate"][0] == parts["value"][0]
        assert parts["query"][0] != parts["gate"][0]
        assert parts["query"][1:] == parts["gate"][1:] == (0, 512)
        assert parts["key"][1:] == parts["value"][1:] == (512, 1024)
        for index, name in enumerate(("query", "key", "gate", "value")):
            format = "q4_g64_fp16" if index < 2 else "q5_g64_fp16"
            _assert_quantized_rows(
                artifact,
                parts[name],
                [(index + 1) * row for row in (1, 2, 4, 8)],
                format,
            )
        assert {
            use["parameter"]: use["activation_policy"]
            for use in artifact.directory.uses
        } == {
            "query": "AllowA4",
            "key": "AllowA4",
            "gate": "AllowA8",
            "value": "AllowA8",
        }


def test_row_overrides_preserve_binding_order_and_coverage(tmp_path):
    model = _model(("weight",))
    model.packing_groups = []
    recipe = Recipe(model)
    recipe.assign("weight", format="q4_g64_fp16", method=grouped_absmax)
    recipe.assign("weight", rows=(1, 3), format="q5_g64_fp16")
    prepared = recipe.prepare(device="cpu", rows_per_chunk=1)
    path = tmp_path / "parts.ninfer"
    _write(path, model, prepared)
    with Artifact(path) as artifact:
        parts = binding_parts(artifact.directory.bindings["weight"], artifact.by_id)
        assert len(parts) == 3
        for part, amplitudes, format in zip(
            parts, ([1], [2, 4], [8]), ("q4_g64_fp16", "q5_g64_fp16", "q4_g64_fp16")
        ):
            _assert_quantized_rows(artifact, part, amplitudes, format)


def test_shared_weight_keeps_use_independent_and_can_be_overridden():
    model = _model(("key", "context_key"))
    model.packing_groups = []
    recipe = Recipe(model)
    recipe.assign("*", format="q8_g32_fp16", method=grouped_absmax)
    recipe.share("context_key", "key")
    recipe.use("context_key", "input", activation_policy="AllowA8")
    shared = recipe.prepare(device="cpu")
    assert len(shared.weights) == 1
    assert shared.bindings["key"] == shared.bindings["context_key"]
    assert shared.uses[0]["activation_policy"] == "A16Only"
    assert shared.uses[1]["activation_policy"] == "AllowA8"
    recipe.assign("context_key", format="q5_g64_fp16")
    independent = recipe.prepare(device="cpu")
    assert len(independent.weights) == 2
    assert independent.bindings["key"] != independent.bindings["context_key"]


def _encoded_source(name, divisor, shift=0):
    codes = (
        (torch.arange(128 * 32) + shift).remainder(256).to(torch.uint8).reshape(128, 32)
    )
    scales = (
        (torch.arange(128 * 4) + shift).remainder(127).to(torch.uint8).reshape(128, 4)
    )
    word = struct.pack("<f", divisor)

    def no_values(begin, end):
        raise AssertionError("encoded import must not require decoded source values")

    return (
        LogicalSource(
            (128, 64),
            name,
            no_values,
            lambda begin, end: EncodedRows(
                "nvfp4", codes[begin:end], scales[begin:end], word
            ),
            lambda: word,
        ),
        codes,
        scales,
    )


def test_nvfp4_encoded_only_import_streams_complete_parent_tiles(tmp_path):
    model = Model({"text": {"config": {}}})
    first, codes, scales = _encoded_source("first", 2.0)
    second, other_codes, other_scales = _encoded_source("second", 2.0, shift=17)
    for name, source in (("gate", first), ("up", second)):
        model.add(Parameter(name, source.shape, source, inputs=("input",)))
    model.packing_groups = [("gate", "up")]
    recipe = Recipe(model)
    recipe.assign("*", format="nvfp4", method=import_encoded)
    recipe.use(
        "gate",
        "input",
        activation_policy="AllowA4",
        auxiliaries={"activation_input_divisor": 1.5},
    )
    prepared = recipe.prepare(device="cpu", rows_per_chunk=128)
    assert len(prepared.weights) == 1 and len(prepared.auxiliaries) == 1
    path = tmp_path / "nvfp4.ninfer"
    _write(path, model, prepared)
    expected = encode_nvfp4(
        torch.cat((codes, other_codes)),
        torch.cat((scales, other_scales)),
        struct.pack("<f", 2.0),
        (256, 64),
    )
    with Artifact(path) as artifact:
        assert artifact.read_object(prepared.weights[0].spec.id) == expected
        assert artifact.read_object(prepared.auxiliaries[0][0].id) == struct.pack(
            "<f", 1.5
        )


def test_incompatible_nvfp4_divisors_keep_default_parents_separate():
    model = Model({"text": {"config": {}}})
    for name, divisor in (("gate", 2.0), ("up", 4.0)):
        source, _, _ = _encoded_source(name, divisor)
        model.add(Parameter(name, source.shape, source))
    model.packing_groups = [("gate", "up")]
    recipe = Recipe(model)
    recipe.assign("*", format="nvfp4", method=import_encoded)
    assert len(recipe.prepare(device="cpu").weights) == 2
    recipe.group(("gate", "up"))
    with pytest.raises(ValueError, match="weight divisors differ"):
        recipe.prepare(device="cpu")


def test_alias_does_not_inherit_another_uses_calibration():
    model = Model({"text": {"config": {}}})
    source, _, _ = _encoded_source("key", 2.0)
    for name in ("key", "context_key"):
        model.add(Parameter(name, source.shape, source, inputs=(name + "/input",)))
    recipe = Recipe(model)
    recipe.share("context_key", "key")
    recipe.assign("key", format="nvfp4", method=import_encoded)
    recipe.use("context_key", "context_key/input", activation_policy="AllowA4")
    with pytest.raises(ValueError, match="independent activation divisor"):
        recipe.prepare(device="cpu")
    with pytest.raises(ValueError, match="positive finite"):
        recipe.use(
            "context_key",
            "context_key/input",
            auxiliaries={"activation_input_divisor": 0.0},
        )
    recipe.use(
        "context_key",
        "context_key/input",
        auxiliaries={"activation_input_divisor": 3.0},
    )
    prepared = recipe.prepare(device="cpu")
    assert len(prepared.weights) == 1
    assert prepared.bindings["key"] == prepared.bindings["context_key"]
    assert prepared.auxiliaries[0][1] == struct.pack("<f", 3.0)


def test_private_component_storage_cannot_be_packed_with_target_weights():
    model = _model(("target", "draft"))
    model.packing_groups = []
    model.parameters["draft"] = replace(model.parameters["draft"], residency="dflash2")
    recipe = Recipe(model)
    recipe.group(("target", "draft"))
    with pytest.raises(ValueError, match="independently selected"):
        recipe.prepare(device="cpu")
    shared = Recipe(model)
    shared.share("draft", "target")
    prepared = shared.prepare(device="cpu")
    assert len(prepared.weights) == 1
    assert prepared.bindings["draft"] == prepared.bindings["target"]
def test_quasar_preserves_nvfp4_casts_qat_controls_and_selects_mtp_source(tmp_path):
    from safetensors.torch import save_file

    quantized_path = tmp_path / "quasar.safetensors"
    tensors = {}
    for prefix, rows in (("projection", 128), ("control", 48)):
        tensors[prefix + ".weight_packed"] = torch.full(
            (rows, 32), 0xA1, dtype=torch.uint8
        )
        tensors[prefix + ".weight_scale"] = torch.full(
            (rows, 4), 0x38, dtype=torch.uint8
        ).view(torch.float8_e4m3fn)
        tensors[prefix + ".weight_global_scale"] = torch.tensor([2.0])
        tensors[prefix + ".input_global_scale"] = torch.tensor([3.0])
    tensors["mtp.weight"] = torch.full((4, 128), 4.0, dtype=torch.bfloat16)
    save_file(tensors, str(quantized_path))
    source = SafetensorsSource(quantized_path)
    model = Model({"text": {"config": {}}, "mtp": {"config": {}, "target": "text"}})
    for name, prefix, shape in (
        ("text/token_embedding", "embedding", (4, 64)),
        ("text/output_head", "head", (4, 64)),
        ("text/layers/0/gdn/query", "projection", (128, 64)),
        ("text/layers/0/gdn/a_projection", "control", (48, 64)),
        ("mtp/input_projection", "mtp", (4, 128)),
    ):
        model.add(
            Parameter(
                name, shape,
                array_source(torch.full(shape, 2.0, dtype=torch.bfloat16), name),
                source_factory=lambda store, format, prefix=prefix, shape=shape: matrix_source(
                    store, prefix + ".weight", shape, format
                ),
                inputs=("input",),
            )
        )
    recipe = Recipe(model)
    qwen3_8_27b_quasar_nvfp4(model, recipe, {"quantized": source})
    prepared = recipe.prepare(device="cpu", rows_per_chunk=16)
    path = tmp_path / "quasar.ninfer"
    _write(path, model, prepared)
    with Artifact(path) as artifact:
        def payload(name):
            parts = binding_parts(artifact.directory.bindings[name], artifact.by_id)
            assert len(parts) == 1
            obj = artifact.object(parts[0][0])
            return obj, artifact.read_object(obj.id)

        projection, encoded = payload("text/layers/0/gdn/query")
        assert projection.format == "nvfp4"
        # Constant source scale words make the exact physical scale-plane expectation independent
        # of production swizzling; neither codes nor stored divisors may be requantized.
        assert encoded == (
            bytes([0xA1]) * (128 * 32)
            + bytes([0x38]) * (128 * 4)
            + struct.pack("<f", 2.0)
        )
        projection_use = next(
            use for use in artifact.directory.uses
            if use["parameter"] == "text/layers/0/gdn/query"
        )
        assert projection_use["activation_policy"] == "AllowA4"
        calibration = projection_use["auxiliaries"]["activation_input_divisor"]
        calibration_parts = binding_parts(calibration, artifact.by_id)
        assert artifact.read_object(calibration_parts[0][0]) == struct.pack("<f", 3.0)
        control, direct = payload("text/layers/0/gdn/a_projection")
        assert control.format == "bf16"
        actual = torch.frombuffer(bytearray(direct), dtype=torch.bfloat16)
        # Signed E2M1 codes 1 and A are +0.5 and -1.0. Stored E4M3FN scale 0x38 is 1;
        # apply the exact divisor 2 before the explicit BF16 cast.
        assert torch.equal(
            actual, torch.tensor([0.25, -0.5] * (48 * 32), dtype=torch.bfloat16)
        )
        for name, value in (
            ("text/token_embedding", 2.0),
            ("text/output_head", 2.0),
            ("mtp/input_projection", 4.0),
        ):
            obj, data = payload(name)
            assert obj.format == "q8_g32_fp16"
            scales, codes = decode_row_split_codes(data, obj.format, obj.shape)
            n, k = obj.shape
            assert bool((codes.reshape(n, -1)[:, :k] == 127).all())
            assert bool(
                (scales[:, : k // 32] == torch.tensor(value / 127, dtype=torch.float16)).all()
            )
