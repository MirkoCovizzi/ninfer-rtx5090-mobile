#pragma once

namespace ninfer::ops::detail {

inline constexpr int kScaleRows    = 128;
inline constexpr int kScaleGroups  = 8;
inline constexpr int kScaleCount   = kScaleRows * kScaleGroups;
inline constexpr int kPaletteSize  = 15;
inline constexpr int kIndexOffset  = 16;
inline constexpr int kIndexBytes   = kScaleCount / 2;
inline constexpr int kPrefixValues = 64;
inline constexpr int kPrefixBytes  = kScaleCount / kPrefixValues * 2;
inline constexpr int kPrefixOffset = kIndexOffset + kIndexBytes;
inline constexpr int kEscapeOffset = kPrefixOffset + kPrefixBytes;

} // namespace ninfer::ops::detail
