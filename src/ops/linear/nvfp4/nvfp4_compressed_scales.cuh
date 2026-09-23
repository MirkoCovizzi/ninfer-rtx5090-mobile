#pragma once

namespace ninfer::ops::detail {

inline constexpr int kScaleRows    = 128;
inline constexpr int kScaleGroups  = 8;
inline constexpr int kScaleCount   = kScaleRows * kScaleGroups;
inline constexpr int kPaletteSize  = 15;
inline constexpr int kIndexBytes   = kScaleCount / 2;
inline constexpr int kPrefixRows   = 8;
inline constexpr int kPrefixBytes  = kScaleRows / kPrefixRows * 2;
inline constexpr int kPrefixOffset = kPaletteSize + kIndexBytes;
inline constexpr int kEscapeOffset = kPrefixOffset + kPrefixBytes;

} // namespace ninfer::ops::detail
