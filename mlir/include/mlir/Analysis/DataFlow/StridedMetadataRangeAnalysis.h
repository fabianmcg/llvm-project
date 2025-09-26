//===- StridedMetadataRange.h - Buffer Range Analysis ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_DATAFLOW_STRIDEDMETADATARANGE_H
#define MLIR_ANALYSIS_DATAFLOW_STRIDEDMETADATARANGE_H

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Interfaces/InferStridedMetadataInterface.h"

namespace mlir {
namespace dataflow {
class StridedMetadataRangeLattice : public Lattice<StridedMetadataRange> {
public:
  using Lattice::Lattice;
};

class StridedMetadataRangeAnalysis
    : public SparseForwardDataFlowAnalysis<StridedMetadataRangeLattice> {
public:
  StridedMetadataRangeAnalysis(DataFlowSolver &solver,
                               int32_t indexBitwidth = 64);

  /// At an entry point, we cannot reason about strided metadata ranges.
  void setToEntryState(StridedMetadataRangeLattice *lattice) override;

  /// Visit an operation. Invoke the transfer function on each operation that
  /// implements ``.
  LogicalResult
  visitOperation(Operation *op,
                 ArrayRef<const StridedMetadataRangeLattice *> operands,
                 ArrayRef<StridedMetadataRangeLattice *> results) override;

private:
  int32_t indexBitwidth = 64;
};
} // namespace dataflow
} // end namespace mlir

#endif // MLIR_ANALYSIS_DATAFLOW_STRIDEDMETADATARANGE_H