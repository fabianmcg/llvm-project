//===- InferStridedMetadataInterface.h - Strided Metadata Inference --*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions of the strided metadata inference interface
// defined in `InferStridedMetadataInterface.td`
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_INFERSTRIDEDMETADATAINTERFACE_H
#define MLIR_INTERFACES_INFERSTRIDEDMETADATAINTERFACE_H

#include "mlir/Interfaces/InferIntRangeInterface.h"

namespace mlir {

/// A class that represents the strided metadata range information for a buffer,
/// including offset, sizes, and strides as integer ranges.
class StridedMetadataRange {
public:
  /// Default constructor creates uninitialized ranges.
  StridedMetadataRange() = default;

  /// Returns a ranked strided metadata range.
  static StridedMetadataRange
  getRanked(ConstantIntRanges offset,
            SmallVectorImpl<ConstantIntRanges> &&sizes,
            SmallVectorImpl<ConstantIntRanges> &&strides) {
    return StridedMetadataRange(offset, std::move(sizes), std::move(strides));
  }

  /// Returns a strided metadata range with maximum ranges.
  static StridedMetadataRange
  getMaxRanges(int32_t indexBitwidth, int32_t sizeRank, int32_t stridedRank) {
    ConstantIntRanges maxValue = ConstantIntRanges::maxRange(indexBitwidth);
    return StridedMetadataRange(
        maxValue, SmallVector<ConstantIntRanges>(sizeRank, maxValue),
        SmallVector<ConstantIntRanges>(stridedRank, maxValue));
  }
  static StridedMetadataRange getMaxRanges(int32_t indexBitwidth,
                                           int32_t rank) {
    return getMaxRanges(indexBitwidth, rank, rank);
  }

  /// Whether the metadata is uninitialized. This happens when the state hasn't
  /// been set during the analysis.
  bool isUninitialized() const { return !offset.has_value(); }

  /// Get the offset range.
  const ConstantIntRanges &getOffset() const {
    assert(offset.has_value() && "Offset range is uninitialized");
    return *offset;
  }

  /// Get the sizes ranges.
  ArrayRef<ConstantIntRanges> getSizes() const { return sizes; }

  /// Get the strides ranges.
  ArrayRef<ConstantIntRanges> getStrides() const { return strides; }

  /// Compare two strided metadata ranges.
  bool operator==(const StridedMetadataRange &other) const {
    return offset == other.offset && sizes == other.sizes &&
           strides == other.strides;
  }

  /// Print the strided metadata range.
  void print(raw_ostream &os) const;

  /// Join two strided metadata ranges, by taking the element-wise union of the
  /// metadata.
  static StridedMetadataRange join(const StridedMetadataRange &lhs,
                                   const StridedMetadataRange &rhs) {
    if (lhs.isUninitialized())
      return rhs;
    if (rhs.isUninitialized())
      return lhs;

    // Helper fuction to compute the range union of constant ranges.
    auto rangeUnion =
        +[](const std::tuple<ConstantIntRanges, ConstantIntRanges> &lhsRhs)
        -> ConstantIntRanges {
      return std::get<0>(lhsRhs).rangeUnion(std::get<1>(lhsRhs));
    };

    // Get the elementwise range union. Note, that `zip_equal` will assert if
    // sizes are not equal.
    auto sizes =
        llvm::map_to_vector(llvm::zip_equal(lhs.sizes, rhs.sizes), rangeUnion);
    auto strides = llvm::map_to_vector(
        llvm::zip_equal(lhs.strides, rhs.strides), rangeUnion);

    // Return the joined metadata.
    return StridedMetadataRange(lhs.offset->rangeUnion(*rhs.offset),
                                std::move(sizes), std::move(strides));
  }

private:
  /// Create a strided metadata range with the given offset, sizes, and strides.
  StridedMetadataRange(ConstantIntRanges offset,
                       SmallVectorImpl<ConstantIntRanges> &&sizes,
                       SmallVectorImpl<ConstantIntRanges> &&strides)
      : offset(std::move(offset)), sizes(std::move(sizes)),
        strides(std::move(strides)) {}

  /// The offset range.
  std::optional<ConstantIntRanges> offset;

  /// The sizes ranges.
  llvm::SmallVector<ConstantIntRanges> sizes;

  /// The strides ranges.
  llvm::SmallVector<ConstantIntRanges> strides;
};

inline raw_ostream &operator<<(raw_ostream &os,
                               const StridedMetadataRange &range) {
  range.print(os);
  return os;
}

using GetIntRangeFn = llvm::function_ref<IntegerValueRange(Value)>;

using SetStridedMetadataRangeFn =
    llvm::function_ref<void(Value, const StridedMetadataRange &)>;
} // end namespace mlir

#include "mlir/Interfaces/InferStridedMetadataInterface.h.inc"

#endif // MLIR_INTERFACES_INFERSTRIDEDMETADATAINTERFACE_H