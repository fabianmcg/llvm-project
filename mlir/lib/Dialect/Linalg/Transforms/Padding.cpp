//===- Padding.cpp - Padding of Linalg ops --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

#define DEBUG_TYPE "linalg-padding"

using namespace mlir;
using namespace mlir::linalg;

#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")

namespace {
/// Helper class for storing padding information.
struct PaddingInfo {
  PaddingInfo(int64_t padToMultipleOf = 1, OpFoldResult size = {})
      : padToMultipleOf(padToMultipleOf), size(size) {}
  bool isTrivial() const { return padToMultipleOf == 1 && size.isNull(); }
  /// Pad the tensor to a multiple of.
  int64_t padToMultipleOf = 1;
  /// The size used for padding.
  OpFoldResult size = {};
};

/// Helper class for storing and computing the padded shape.
struct PaddedShape {
  /// Initializes the shape information and returns whether the shape of the
  /// operand will change.
  bool initialize(linalg::LinalgOp opToPad, OpOperand *opOperand,
                  const LinalgPaddingOptions &options);

  /// Computs the padded shape.
  void computePadding(OpBuilder &builder, Value operand);

  /// Returns the new tensor type.
  RankedTensorType getType(Type elemTy) {
    return RankedTensorType::get(shape, elemTy);
  }

  /// Return the dynamic dimensions of the shape.
  ValueRange getDynamicDims() { return dynDims; }

private:
  SmallVector<int64_t> shape;
  SmallVector<Value> dynDims;
  DenseMap<int64_t, PaddingInfo> dimToInfo;
};
} // namespace

bool PaddedShape::initialize(linalg::LinalgOp opToPad, OpOperand *opOperand,
                             const LinalgPaddingOptions &options) {
  AffineMap indexingMap = opToPad.getMatchingIndexingMap(opOperand);

  // Initialize the padded shape.
  llvm::append_range(shape, opToPad.getShape(opOperand));

  // Return early if there's no padding involved.
  if (!options.padToMultipleOf && !options.padToSizeOf)
    return true;

  // Collect the shape dimensions that are a function of "paddingDimensions",
  // along with the multiple that they should be padded to ("1" if none).
  bool alreadyHasRequestedShape = true;
  for (const auto [dimIndex, dim] : enumerate(options.paddingDimensions)) {
    for (const auto [exprIndex, expr] : enumerate(indexingMap.getResults())) {
      if (expr.isFunctionOfDim(dim)) {
        int64_t dimSize = shape[exprIndex];
        PaddingInfo paddingInfo;
        if (options.padToMultipleOf.has_value())
          paddingInfo.padToMultipleOf = (*options.padToMultipleOf)[dimIndex];
        if (options.padToSizeOf.has_value())
          paddingInfo.size = (*options.padToSizeOf)[dimIndex];

        // Skip if the padding information is trivial. Note that dynamic
        // dimensions never have trivial padding information.
        if (paddingInfo.isTrivial() && !ShapedType::isDynamic(dimSize))
          continue;
        dimToInfo[exprIndex] = paddingInfo;
        if (ShapedType::isDynamic(dimSize) ||
            dimSize % paddingInfo.padToMultipleOf != 0) {
          alreadyHasRequestedShape = false;
        }
      }
    }
  }

  return alreadyHasRequestedShape;
}

void PaddedShape::computePadding(OpBuilder &builder, Value operand) {
  Location loc = operand.getLoc();
  AffineExpr sizeSym = builder.getAffineSymbolExpr(0);

  // Compute the padding for each dimension.
  for (auto &&[i, dim] : llvm::enumerate(shape)) {
    LLVM_DEBUG(DBGS() << "--compute padded size for dim " << i << "\n");

    // Skip dimensions that do not require padding.
    if (!dimToInfo.contains(i)) {
      LLVM_DEBUG(DBGS() << "----dim does not require padding, SKIP\n");
      if (ShapedType::isDynamic(dim)) {
        dynDims.push_back(
            cast<Value>(tensor::getMixedSize(builder, loc, operand, i)));
      }
      continue;
    }
    PaddingInfo paddingInfo = dimToInfo[i];
    OpFoldResult size = paddingInfo.size;
    // Get the tensor dim size if none was provided.
    if (size.isNull())
      size = tensor::getMixedSize(builder, loc, operand, i);

    // Compute the padded size to be a multiple of `padToMultipleOf`.
    AffineExpr szExpr = (sizeSym).ceilDiv(paddingInfo.padToMultipleOf) *
                        paddingInfo.padToMultipleOf;
    OpFoldResult paddedSize =
        affine::makeComposedFoldedAffineApply(builder, loc, szExpr, size);
    assert(paddedSize && "invalid arguments to affine apply");
    if (auto cstSzAttr = dyn_cast<Attribute>(paddedSize)) {
      dim = cast<IntegerAttr>(cstSzAttr).getValue().getZExtValue();
    } else {
      dim = ShapedType::kDynamic;
      dynDims.push_back(cast<Value>(paddedSize));
    }
    LLVM_DEBUG(DBGS() << "----new dim size: " << paddedSize << "\n");
  }
}

/// Pad the `opOperand` in the "paddingDimensions" using the padding value and
/// the nofold flag found in "paddingValues" and "nofoldFlags", respectively.
///
/// Exit early and return the `opOperand` value if it already has the requested
/// shape. i.e.:
/// - static shape
/// - nofold is not set
/// - dim sizes are multiples of "padToMultipleOf"
///
/// Otherwise, try to pad the shape dimensions that match the iterator
/// dimensions "paddingDimensions" and return the tensor::PadOp result if
/// padding succeeds or failure otherwise.
static FailureOr<Value> padOperandToSmallestStaticBoundingBox(
    RewriterBase &rewriter, linalg::LinalgOp opToPad, OpOperand *opOperand,
    const LinalgPaddingOptions &options) {
  assert(
      (!options.padToMultipleOf.has_value() ||
       options.padToMultipleOf->size() == options.paddingDimensions.size()) &&
      "invalid number of elements in padToMultipleOf");

  // Initialize the padded shape.
  PaddedShape shape;
  bool alreadyHasRequestedShape = shape.initialize(opToPad, opOperand, options);

  // Return the unpadded operand if padding to a static shape is not needed and
  // if the nofold flag is not set.
  bool nofold = opOperand->getOperandNumber() < options.nofoldFlags.size()
                    ? bool(options.nofoldFlags[opOperand->getOperandNumber()])
                    : false;
  if (!nofold && alreadyHasRequestedShape)
    return opOperand->get();

  // Fail if `paddingValues` specifies no padding value.
  if (opOperand->getOperandNumber() >= options.paddingValues.size()) {
    return rewriter.notifyMatchFailure(opToPad, "--no padding value specified");
  }
  Attribute paddingAttr = options.paddingValues[opOperand->getOperandNumber()];

  Value paddingValue;
  if (auto complexTy = dyn_cast<ComplexType>(
          getElementTypeOrSelf(opOperand->get().getType()))) {
    auto complexAttr = cast<ArrayAttr>(paddingAttr);
    paddingValue = rewriter.create<complex::ConstantOp>(opToPad.getLoc(),
                                                        complexTy, complexAttr);
  } else {
    paddingValue = rewriter.create<arith::ConstantOp>(
        opToPad.getLoc(), cast<TypedAttr>(paddingAttr));
  }

  // If needed, the padding for each dimension.
  if (!alreadyHasRequestedShape)
    shape.computePadding(rewriter, opOperand->get());

  // Compute the new tensor type.
  RankedTensorType paddedTensorType =
      shape.getType(getElementTypeOrSelf(opOperand->get()));
  LLVM_DEBUG(DBGS() << "--SUCCESS, makeComposedPadHighOp with type: "
                    << paddedTensorType << "\n");

  // Pad the operand to the bounding box defined by `shape`.
  return makeComposedPadHighOp(rewriter, opToPad->getLoc(), paddedTensorType,
                               opOperand->get(), paddingValue, nofold,
                               shape.getDynamicDims());
}

LogicalResult
linalg::rewriteAsPaddedOp(RewriterBase &rewriter, LinalgOp opToPad,
                          const LinalgPaddingOptions &constOptions,
                          LinalgOp &paddedOp, SmallVector<Value> &replacements,
                          SmallVector<tensor::PadOp> &padOps) {
  LLVM_DEBUG(DBGS() << "Start rewriteAsPaddedOp : " << opToPad << "\n");
  Location loc = opToPad->getLoc();

  LinalgPaddingOptions options(constOptions);
  // Allow inference of pad values if they are not explicitly specified.
  // TODO: be mindful about the value depending on the actual operation.
  if (options.paddingValues.empty()) {
    SmallVector<Type> types(opToPad->getOperandTypes());
    llvm::append_range(types, opToPad->getResultTypes());
    for (Type t : types) {
      options.paddingValues.push_back(
          rewriter.getZeroAttr(getElementTypeOrSelf(t)));
    }
  }

  // TODO: there are cases where we may still want to pad to larger sizes.
  if (!opToPad.hasPureTensorSemantics())
    return rewriter.notifyMatchFailure(opToPad,
                                       "expected operation on tensors");

  OpBuilder::InsertionGuard g(rewriter);
  // Set IP after op because we also take the dims of the original output.
  rewriter.setInsertionPointAfter(opToPad);

  // Make a copy of the shaped operands and update it.
  SmallVector<Value> newOperands;
  newOperands.reserve(opToPad->getNumOperands());
  for (OpOperand &opOperand : opToPad->getOpOperands()) {
    FailureOr<Value> paddedOperand = padOperandToSmallestStaticBoundingBox(
        rewriter, opToPad, &opOperand, options);
    // Exit if `paddingDimensions` cannot be bounded statically.
    if (failed(paddedOperand)) {
      LLVM_DEBUG(DBGS() << "--operand cannot be bound statically : "
                        << opOperand.get() << " -> FAIL\n");
      return rewriter.notifyMatchFailure(opToPad,
                                         "operand cannot be bound statically");
    }
    newOperands.push_back(*paddedOperand);
    if (auto padOp = paddedOperand->getDefiningOp<tensor::PadOp>())
      padOps.push_back(padOp);
  }

  ReifiedRankedShapedTypeDims reifiedResultShapes;
  if (failed(reifyResultShapes(rewriter, opToPad, reifiedResultShapes))) {
    LLVM_DEBUG(DBGS() << "--failed to reify result shapes -> FAIL\n");
    return rewriter.notifyMatchFailure(opToPad,
                                       "failed to reify result shapes");
  }
  assert(reifiedResultShapes.size() == opToPad->getNumResults() &&
         "expected same number of results");

  // Clone `opToPad` to operate on the statically padded shapes.
  auto resultTensorTypes =
      ValueRange(newOperands).take_back(opToPad.getNumDpsInits()).getTypes();
  // clone **should** properly notify the rewriter.
  paddedOp = clone(rewriter, opToPad, resultTensorTypes, newOperands);
  LLVM_DEBUG(DBGS() << "--cloned padded op: " << paddedOp << "\n");

  // Recover the slice out of the new static results. This keeps the original
  // linalg op around because it uses the dims of the original results.
  SmallVector<Value> paddedSubtensorResults;
  paddedSubtensorResults.reserve(opToPad->getNumResults());
  for (const auto &en : llvm::enumerate(paddedOp->getResults())) {
    Value paddedResult = en.value();
    int64_t resultNumber = en.index();
    int64_t rank = cast<RankedTensorType>(paddedResult.getType()).getRank();
    SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
    paddedSubtensorResults.push_back(rewriter.create<tensor::ExtractSliceOp>(
        loc, paddedResult, offsets, reifiedResultShapes[resultNumber],
        strides));
  }

  if (options.copyBackOp == LinalgPaddingOptions::CopyBackOp::None) {
    replacements = std::move(paddedSubtensorResults);
    return success();
  }

  // Copy back unpadded results to the original destination (i.e., inits of the
  // linalg op), so that the destination buffer of the computation does not
  // change. If the padding folds away, this will materialize as a memcpy
  // between two identical buffers, which will then also fold away.
  assert(static_cast<int64_t>(paddedSubtensorResults.size()) ==
             opToPad.getNumDpsInits() &&
         "expected matching number of results");
  for (auto it :
       llvm::zip(paddedSubtensorResults, opToPad.getDpsInitsMutable())) {
    if (options.copyBackOp == LinalgPaddingOptions::CopyBackOp::LinalgCopy) {
      replacements.push_back(rewriter
                                 .create<linalg::CopyOp>(loc, std::get<0>(it),
                                                         std::get<1>(it).get())
                                 .getResult(0));
    } else if (options.copyBackOp ==
               LinalgPaddingOptions::CopyBackOp::
                   BufferizationMaterializeInDestination) {
      replacements.push_back(
          rewriter
              .create<bufferization::MaterializeInDestinationOp>(
                  loc, std::get<0>(it), std::get<1>(it).get())
              ->getResult(0));
    } else {
      llvm_unreachable("unsupported copy back op");
    }
  }
  return success();
}

FailureOr<LinalgOp>
mlir::linalg::padAndHoistLinalgOp(RewriterBase &rewriter, LinalgOp linalgOp,
                                  const LinalgPaddingOptions &options) {
  assert(options.copyBackOp == LinalgPaddingOptions::CopyBackOp::None &&
         "invalid options");

  if (!linalgOp.hasPureTensorSemantics())
    return rewriter.notifyMatchFailure(
        linalgOp, "only applies to Linalg ops with tensor semantics");

  // Pad the operation.
  LinalgOp paddedOp;
  SmallVector<Value> newResults;
  SmallVector<tensor::PadOp> padOps;
  if (failed(rewriteAsPaddedOp(rewriter, linalgOp, options, paddedOp,
                               newResults, padOps)))
    return rewriter.notifyMatchFailure(linalgOp,
                                       "failed to rewrite as a padded op");

  // Hoist the padding.
  for (const auto &en : enumerate(options.hoistPaddings)) {
    if (static_cast<int64_t>(en.index()) >= paddedOp->getNumOperands())
      break;
    OpOperand &opOperand = paddedOp->getOpOperand(en.index());
    auto padOp = opOperand.get().getDefiningOp<tensor::PadOp>();
    if (!padOp || en.value() == 0) {
      (void)rewriter.notifyMatchFailure(linalgOp, "not a tensor.pad -- skip");
      continue;
    }

    // Fail hoisting if the operand shape is not fully static.
    if (llvm::any_of(paddedOp.getShape(&opOperand), ShapedType::isDynamic)) {
      (void)rewriter.notifyMatchFailure(linalgOp,
                                        "non static padding shape -- skip");
      continue;
    }

    tensor::PadOp hoistedOp;
    SmallVector<TransposeOp> transposeOps;
    SmallVector<int64_t> transposeVector =
        en.index() < options.transposePaddings.size()
            ? options.transposePaddings[en.index()]
            : SmallVector<int64_t>{};

    FailureOr<Value> newResult = hoistPaddingOnTensors(
        padOp, en.value(), transposeVector, hoistedOp, transposeOps);
    if (failed(newResult)) {
      (void)rewriter.notifyMatchFailure(linalgOp,
                                        "failed to apply hoistPadding");
      continue;
    }
    rewriter.replaceOp(padOp, *newResult);
  }

  // Replace the original operation to pad.
  rewriter.replaceOp(linalgOp, newResults);

  return paddedOp;
}
