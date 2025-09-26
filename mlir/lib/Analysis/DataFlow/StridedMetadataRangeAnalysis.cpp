//===- StridedMetadataRangeAnalysis.cpp - Integer range analysis --------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the dataflow analysis class for integer range inference
// which is used in transformations over the `arith` dialect such as
// branch elimination or signed->unsigned rewriting
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlow/StridedMetadataRangeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/DebugStringHelper.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "strided-metadata-range-analysis"

using namespace mlir;
using namespace mlir::dataflow;

static StridedMetadataRange getEntryStateImpl(Value v, int32_t indexBitwidth) {
  auto mTy = dyn_cast<BaseMemRefType>(v.getType());
  if (!mTy || !mTy.hasRank())
    return StridedMetadataRange::getMaxRanges(0, 0);
  return StridedMetadataRange::getMaxRanges(indexBitwidth, mTy.getRank());
}

StridedMetadataRangeAnalysis::StridedMetadataRangeAnalysis(
    DataFlowSolver &solver, int32_t indexBitwidth)
    : SparseForwardDataFlowAnalysis(solver), indexBitwidth(indexBitwidth) {
  assert(indexBitwidth > 0 && "invalid bitwidth");
}

void StridedMetadataRangeAnalysis::setToEntryState(
    StridedMetadataRangeLattice *lattice) {
  propagateIfChanged(lattice, lattice->join(getEntryStateImpl(
                                  lattice->getAnchor(), indexBitwidth)));
}

LogicalResult StridedMetadataRangeAnalysis::visitOperation(
    Operation *op, ArrayRef<const StridedMetadataRangeLattice *> operands,
    ArrayRef<StridedMetadataRangeLattice *> results) {
  auto inferrable = dyn_cast<InferStridedMetadataInterface>(op);
  if (!inferrable) {
    setAllToEntryStates(results);
    return success();
  }
  auto getIntRange = [&](Value value) -> IntegerValueRange {
    auto lattice = getOrCreateFor<IntegerValueRangeLattice>(
        getProgramPointAfter(op), value);
    return lattice ? lattice->getValue() : IntegerValueRange();
  };

  LDBG() << "Inferring metadata for "
         << OpWithFlags(op, OpPrintingFlags().skipRegions());
  auto argRanges = llvm::map_to_vector(
      operands, [](const StridedMetadataRangeLattice *lattice) {
        return lattice->getValue();
      });

  auto joinCallback = [&](Value v, const StridedMetadataRange &md) {
    auto result = cast<OpResult>(v);
    assert(llvm::is_contained(op->getResults(), result));
    LDBG() << "Inferred metadata " << md;
    StridedMetadataRangeLattice *lattice = results[result.getResultNumber()];
    propagateIfChanged(lattice, lattice->join(md));
  };
  inferrable.inferStridedMetadataRanges(argRanges, getIntRange, joinCallback,
                                        indexBitwidth);
  return success();
}
