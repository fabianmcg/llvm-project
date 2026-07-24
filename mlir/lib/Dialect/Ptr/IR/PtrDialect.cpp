//===- PtrDialect.cpp - Pointer dialect ---------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Pointer dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Ptr/IR/MemorySpaceInterfaces.h"
#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/Dialect/Ptr/IR/PtrTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include <utility>

using namespace mlir;
using namespace mlir::ptr;

//===----------------------------------------------------------------------===//
// Pointer dialect
//===----------------------------------------------------------------------===//

void PtrDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Ptr/IR/PtrOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/Ptr/IR/PtrOpsAttrs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/Ptr/IR/PtrOpsTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Common helper functions.
//===----------------------------------------------------------------------===//

/// Verifies that the alignment attribute is a power of 2 if present.
static LogicalResult
verifyAlignment(std::optional<int64_t> alignment,
                function_ref<InFlightDiagnostic()> emitError) {
  if (!alignment)
    return success();
  if (alignment.value() <= 0)
    return emitError() << "alignment must be positive";
  if (!llvm::isPowerOf2_64(alignment.value()))
    return emitError() << "alignment must be a power of 2";
  return success();
}

enum class MaskFormat { AllTrue = 0, AllFalse = 1, Unknown = 2 };

/// Inspects a constant dense i1 attribute to classify the mask as all-true,
/// all-false, or unknown. Returns Unknown for dynamic or mixed masks.
static MaskFormat getMaskFormat(Value mask) {
  DenseIntElementsAttr denseElts;
  if (!matchPattern(mask, m_Constant(&denseElts)))
    return MaskFormat::Unknown;
  int64_t val = 0;
  for (bool b : denseElts.getValues<bool>()) {
    if (b && val >= 0) {
      val++;
      continue;
    }
    if (!b && val <= 0) {
      val--;
      continue;
    }
    return MaskFormat::Unknown;
  }
  if (val > 0)
    return MaskFormat::AllTrue;
  if (val < 0)
    return MaskFormat::AllFalse;
  return MaskFormat::Unknown;
}

//===----------------------------------------------------------------------===//
// FutureVerifierOpTrait implementation
//===----------------------------------------------------------------------===//

LogicalResult ptr::impl::verifyFutureTrait(Operation *op, PtrType ptrType,
                                           FutureKind expectedKind,
                                           Value futureValue) {
  if (!futureValue)
    return success();
  auto futureType = dyn_cast<FutureType>(futureValue.getType());
  if (!futureType)
    return success();
  if (futureType.getMemorySpace() != ptrType.getMemorySpace())
    return op->emitOpError(
        "future memory space does not match pointer memory space");
  // Opaque futures are compatible with any expected kind.
  if (futureType.getKind() == FutureKind::opaque)
    return success();
  if (futureType.getKind() != expectedKind)
    return op->emitOpError("future kind does not match operation kind");
  return success();
}

//===----------------------------------------------------------------------===//
// CastToOpaqueOp
//===----------------------------------------------------------------------===//

namespace {
struct FoldCastToOpaqueOp : public OpRewritePattern<CastToOpaqueOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(CastToOpaqueOp op,
                                PatternRewriter &rewriter) const override {
    if (cast<FutureType>(op.getFuture().getType()).getKind() !=
        FutureKind::opaque)
      return failure();
    rewriter.replaceOp(op, op.getFuture());
    return success();
  }
};
} // namespace

void CastToOpaqueOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  results.add<FoldCastToOpaqueOp>(context);
}

LogicalResult CastToOpaqueOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, PropertyRef properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto futureType = cast<FutureType>(operands[0].getType());
  inferredReturnTypes.push_back(FutureType::get(futureType.getMemorySpace(),
                                                FutureKind::opaque,
                                                futureType.getElementType()));
  return success();
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) { return getValue(); }

//===----------------------------------------------------------------------===//
// FromPtrOp
//===----------------------------------------------------------------------===//

OpFoldResult FromPtrOp::fold(FoldAdaptor adaptor) {
  // Fold the pattern:
  // %ptr = ptr.to_ptr %v : type -> ptr
  // (%mda = ptr.get_metadata %v : type)?
  // %val = ptr.from_ptr %ptr (metadata %mda)? : ptr -> type
  // To:
  // %val -> %v
  Value ptrLike;
  FromPtrOp fromPtr = *this;
  while (fromPtr != nullptr) {
    auto toPtr = fromPtr.getPtr().getDefiningOp<ToPtrOp>();
    // Cannot fold if it's not a `to_ptr` op or the initial and final types are
    // different.
    if (!toPtr || toPtr.getPtr().getType() != fromPtr.getType())
      return ptrLike;
    Value md = fromPtr.getMetadata();
    // If the type has trivial metadata fold.
    if (!fromPtr.getType().hasPtrMetadata()) {
      ptrLike = toPtr.getPtr();
    } else if (md) {
      // Fold if the metadata can be verified to be equal.
      if (auto mdOp = md.getDefiningOp<GetMetadataOp>();
          mdOp && mdOp.getPtr() == toPtr.getPtr())
        ptrLike = toPtr.getPtr();
    }
    // Check for a sequence of casts.
    fromPtr = ptrLike ? ptrLike.getDefiningOp<FromPtrOp>() : nullptr;
  }
  return ptrLike;
}

LogicalResult FromPtrOp::verify() {
  if (isa<PtrType>(getType()))
    return emitError() << "the result type cannot be `!ptr.ptr`";
  if (getType().getMemorySpace() != getPtr().getType().getMemorySpace()) {
    return emitError()
           << "expected the input and output to have the same memory space";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

void GatherOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // Gather performs reads from multiple memory locations specified by ptrs
  effects.emplace_back(MemoryEffects::Read::get(), &getPtrsMutable());
}

LogicalResult GatherOp::verify() {
  auto emitDiag = [&]() -> InFlightDiagnostic { return emitError(); };

  // Verify that the pointer type's memory space allows loads.
  MemorySpaceAttrInterface ms =
      cast<PtrType>(getPtrs().getType().getElementType()).getMemorySpace();
  DataLayout dataLayout = DataLayout::closest(*this);
  if (!ms.isValidLoad(getResult().getType(), AtomicOrdering::not_atomic,
                      getAlignment(), &dataLayout, emitDiag))
    return failure();

  // Verify the alignment.
  return verifyAlignment(getAlignment(), emitDiag);
}

void GatherOp::build(OpBuilder &builder, OperationState &state, Type resultType,
                     Value ptrs, Value mask, Value passthrough,
                     unsigned alignment) {
  build(builder, state, resultType, ptrs, mask, passthrough,
        alignment ? std::optional<int64_t>(alignment) : std::nullopt);
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

/// Verifies the attributes and the type of atomic memory access operations.
template <typename OpTy>
static LogicalResult
verifyAtomicMemOp(OpTy memOp, ArrayRef<AtomicOrdering> unsupportedOrderings) {
  if (memOp.getOrdering() != AtomicOrdering::not_atomic) {
    if (llvm::is_contained(unsupportedOrderings, memOp.getOrdering()))
      return memOp.emitOpError("unsupported ordering '")
             << stringifyAtomicOrdering(memOp.getOrdering()) << "'";
    if (!memOp.getAlignment())
      return memOp.emitOpError("expected alignment for atomic access");
    return success();
  }
  if (memOp.getSyncscope()) {
    return memOp.emitOpError(
        "expected syncscope to be null for non-atomic access");
  }
  return success();
}

void LoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getPtrMutable());
  // Volatile operations can have target-specific read-write effects on
  // memory besides the one referred to by the pointer operand.
  // Similarly, atomic operations that are monotonic or stricter cause
  // synchronization that from a language point-of-view, are arbitrary
  // read-writes into memory.
  if (getVolatile_() || (getOrdering() != AtomicOrdering::not_atomic &&
                         getOrdering() != AtomicOrdering::unordered)) {
    effects.emplace_back(MemoryEffects::Write::get());
    effects.emplace_back(MemoryEffects::Read::get());
  }
}

LogicalResult LoadOp::verify() {
  auto emitDiag = [&]() -> InFlightDiagnostic { return emitError(); };
  MemorySpaceAttrInterface ms = getPtr().getType().getMemorySpace();
  DataLayout dataLayout = DataLayout::closest(*this);
  if (!ms.isValidLoad(getResult().getType(), getOrdering(), getAlignment(),
                      &dataLayout, emitDiag))
    return failure();
  if (failed(verifyAlignment(getAlignment(), emitDiag)))
    return failure();
  return verifyAtomicMemOp(*this,
                           {AtomicOrdering::release, AtomicOrdering::acq_rel});
}

void LoadOp::build(OpBuilder &builder, OperationState &state, Type type,
                   Value addr, unsigned alignment, bool isVolatile,
                   bool isNonTemporal, bool isInvariant, bool isInvariantGroup,
                   AtomicOrdering ordering, StringRef syncscope) {
  build(builder, state, type, addr,
        alignment ? std::optional<int64_t>(alignment) : std::nullopt,
        isVolatile, isNonTemporal, isInvariant, isInvariantGroup, ordering,
        syncscope.empty() ? nullptr : builder.getStringAttr(syncscope));
}
//===----------------------------------------------------------------------===//
// MaskedLoadOp
//===----------------------------------------------------------------------===//

void MaskedLoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // MaskedLoad performs reads from the memory location specified by ptr.
  effects.emplace_back(MemoryEffects::Read::get(), &getPtrMutable());
}

LogicalResult MaskedLoadOp::verify() {
  auto emitDiag = [&]() -> InFlightDiagnostic { return emitError(); };
  // Verify that the pointer type's memory space allows loads.
  MemorySpaceAttrInterface ms = getPtr().getType().getMemorySpace();
  DataLayout dataLayout = DataLayout::closest(*this);
  if (!ms.isValidLoad(getResult().getType(), AtomicOrdering::not_atomic,
                      getAlignment(), &dataLayout, emitDiag))
    return failure();

  // Verify the alignment.
  return verifyAlignment(getAlignment(), emitDiag);
}

void MaskedLoadOp::build(OpBuilder &builder, OperationState &state,
                         Type resultType, Value ptr, Value mask,
                         Value passthrough, unsigned alignment) {
  build(builder, state, resultType, ptr, mask, passthrough,
        alignment ? std::optional<int64_t>(alignment) : std::nullopt);
}

struct MaskedLoadFolder : public OpRewritePattern<MaskedLoadOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(MaskedLoadOp load,
                                PatternRewriter &rewriter) const override {
    switch (getMaskFormat(load.getMask())) {
    case MaskFormat::AllTrue:
      rewriter.replaceOpWithNewOp<LoadOp>(
          load, load.getType(), load.getPtr(),
          static_cast<unsigned>(load.getAlignment().value_or(0)));
      return success();
    case MaskFormat::AllFalse:
      rewriter.replaceOp(load, load.getPassthrough());
      return success();
    case MaskFormat::Unknown:
      return failure();
    }
    llvm_unreachable("Unexpected MaskFormat on MaskedLoad.");
  }
};

void MaskedLoadOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.add<MaskedLoadFolder>(context);
}

//===----------------------------------------------------------------------===//
// MaskedStoreOp
//===----------------------------------------------------------------------===//

void MaskedStoreOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // MaskedStore performs writes to the memory location specified by ptr
  effects.emplace_back(MemoryEffects::Write::get(), &getPtrMutable());
}

LogicalResult MaskedStoreOp::verify() {
  auto emitDiag = [&]() -> InFlightDiagnostic { return emitError(); };
  // Verify that the pointer type's memory space allows stores.
  MemorySpaceAttrInterface ms = getPtr().getType().getMemorySpace();
  DataLayout dataLayout = DataLayout::closest(*this);
  if (!ms.isValidStore(getValue().getType(), AtomicOrdering::not_atomic,
                       getAlignment(), &dataLayout, emitDiag))
    return failure();

  // Verify the alignment.
  return verifyAlignment(getAlignment(), emitDiag);
}

void MaskedStoreOp::build(OpBuilder &builder, OperationState &state,
                          Value value, Value ptr, Value mask,
                          unsigned alignment, bool hasFuture) {
  Type futureType;
  if (hasFuture)
    futureType = FutureType::get(cast<PtrType>(ptr.getType()).getMemorySpace(),
                                 FutureKind::write);
  build(builder, state, futureType, value, ptr, mask,
        alignment ? std::optional<int64_t>(alignment) : std::nullopt);
}

namespace {
struct MaskedStoreFolder : public OpRewritePattern<MaskedStoreOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(MaskedStoreOp store,
                                PatternRewriter &rewriter) const override {
    switch (getMaskFormat(store.getMask())) {
    case MaskFormat::AllTrue:
      rewriter.replaceOpWithNewOp<StoreOp>(
          store, store.getValue(), store.getPtr(),
          static_cast<unsigned>(store.getAlignment().value_or(0)),
          /*isVolatile=*/false,
          /*isNonTemporal=*/false,
          /*isInvariantGroup=*/false,
          /*ordering=*/AtomicOrdering::not_atomic,
          /*syncscope=*/StringRef(),
          /*hasFuture=*/static_cast<bool>(store.getFuture()));
      return success();
    case MaskFormat::AllFalse:
      if (store.getFuture() && !store.getFuture().use_empty())
        return failure();
      rewriter.eraseOp(store);
      return success();
    case MaskFormat::Unknown:
      return failure();
    }
    llvm_unreachable("Unexpected MaskFormat on MaskedStore.");
  }
};
} // namespace

void MaskedStoreOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  results.add<MaskedStoreFolder>(context);
}

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//

void ScatterOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // Scatter performs writes to multiple memory locations specified by ptrs
  effects.emplace_back(MemoryEffects::Write::get(), &getPtrsMutable());
}

LogicalResult ScatterOp::verify() {
  auto emitDiag = [&]() -> InFlightDiagnostic { return emitError(); };

  // Verify that the pointer type's memory space allows stores.
  MemorySpaceAttrInterface ms =
      cast<PtrType>(getPtrs().getType().getElementType()).getMemorySpace();
  DataLayout dataLayout = DataLayout::closest(*this);
  if (!ms.isValidStore(getValue().getType(), AtomicOrdering::not_atomic,
                       getAlignment(), &dataLayout, emitDiag))
    return failure();

  // Verify the alignment.
  return verifyAlignment(getAlignment(), emitDiag);
}

void ScatterOp::build(OpBuilder &builder, OperationState &state, Value value,
                      Value ptrs, Value mask, unsigned alignment,
                      bool hasFuture) {
  Type futureType;
  if (hasFuture) {
    MemorySpaceAttrInterface ms =
        cast<PtrType>(cast<ShapedType>(ptrs.getType()).getElementType())
            .getMemorySpace();
    futureType = FutureType::get(ms, FutureKind::write);
  }
  build(builder, state, futureType, value, ptrs, mask,
        alignment ? std::optional<int64_t>(alignment) : std::nullopt);
}

//===----------------------------------------------------------------------===//
// WaitOp
//===----------------------------------------------------------------------===//

LogicalResult
WaitOp::inferReturnTypes(MLIRContext *context, std::optional<Location> location,
                         ValueRange operands, DictionaryAttr attributes,
                         PropertyRef properties, RegionRange regions,
                         SmallVectorImpl<Type> &inferredReturnTypes) {
  for (Value future : operands) {
    Type elementType = cast<FutureType>(future.getType()).getElementType();
    if (elementType)
      inferredReturnTypes.push_back(elementType);
  }
  return success();
}

LogicalResult WaitOp::verify() {
  for (Type ty : getFences()) {
    auto futureType = dyn_cast<FutureType>(ty);
    if (!futureType || futureType.getElementType())
      return emitError("fences must contain only empty future types");
  }
  return success();
}

LogicalResult WaitOp::canonicalize(WaitOp op,
                                   ::mlir::PatternRewriter &rewriter) {
  ArrayRef<FutureType> fences = op.getFences();
  if (fences.empty())
    return failure();
  SmallVector<FutureType> newFences;

  // Compute the unique fences in stable order, while also promoting read/write
  // fences to opaque fences if they both appear.
  {
    DenseMap<MemorySpaceAttrInterface, int32_t> fenceKinds;
    for (FutureType fence : fences) {
      int32_t &kinds = fenceKinds[fence.getMemorySpace()];
      switch (fence.getKind()) {
      case FutureKind::opaque:
        kinds |= 1;
        break;
      case FutureKind::read:
        kinds |= 2;
        break;
      case FutureKind::write:
        kinds |= 4;
        break;
      }
    }

    DenseSet<FutureType> uniqueFences;
    for (FutureType fence : fences) {
      int32_t kinds = fenceKinds[fence.getMemorySpace()];
      if ((kinds & 1) == 1 || (kinds & 6) == 6)
        fence = FutureType::get(fence.getMemorySpace(), FutureKind::opaque);
      if (!uniqueFences.insert(fence).second)
        continue;
      newFences.push_back(fence);
    }
  }

  // Sort the fences by kind, so that opaque fences are at the beginning of the
  // list.
  llvm::stable_sort(newFences, [](FutureType a, FutureType b) {
    return a.getKind() < b.getKind();
  });

  // If the new fences are the same as the old fences, return.
  if (fences == ArrayRef<FutureType>(newFences))
    return failure();

  rewriter.modifyOpInPlace(
      op, [&]() { op.getProperties().fences = std::move(newFences); });
  return success();
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

void StoreOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getPtrMutable());
  // Volatile operations can have target-specific read-write effects on
  // memory besides the one referred to by the pointer operand.
  // Similarly, atomic operations that are monotonic or stricter cause
  // synchronization that from a language point-of-view, are arbitrary
  // read-writes into memory.
  if (getVolatile_() || (getOrdering() != AtomicOrdering::not_atomic &&
                         getOrdering() != AtomicOrdering::unordered)) {
    effects.emplace_back(MemoryEffects::Write::get());
    effects.emplace_back(MemoryEffects::Read::get());
  }
}

LogicalResult StoreOp::verify() {
  auto emitDiag = [&]() -> InFlightDiagnostic { return emitError(); };
  MemorySpaceAttrInterface ms = getPtr().getType().getMemorySpace();
  DataLayout dataLayout = DataLayout::closest(*this);
  if (!ms.isValidStore(getValue().getType(), getOrdering(), getAlignment(),
                       &dataLayout, emitDiag))
    return failure();
  if (failed(verifyAlignment(getAlignment(), emitDiag)))
    return failure();
  return verifyAtomicMemOp(*this,
                           {AtomicOrdering::acquire, AtomicOrdering::acq_rel});
}

void StoreOp::build(OpBuilder &builder, OperationState &state, Value value,
                    Value addr, unsigned alignment, bool isVolatile,
                    bool isNonTemporal, bool isInvariantGroup,
                    AtomicOrdering ordering, StringRef syncscope,
                    bool hasFuture) {
  Type futureType;
  if (hasFuture)
    futureType = FutureType::get(cast<PtrType>(addr.getType()).getMemorySpace(),
                                 FutureKind::write);
  build(builder, state, futureType, value, addr,
        alignment ? std::optional<int64_t>(alignment) : std::nullopt,
        isVolatile, isNonTemporal, isInvariantGroup, ordering,
        syncscope.empty() ? nullptr : builder.getStringAttr(syncscope));
}

//===----------------------------------------------------------------------===//
// PtrAddOp
//===----------------------------------------------------------------------===//

/// Fold: ptradd ptr + 0 ->  ptr
OpFoldResult PtrAddOp::fold(FoldAdaptor adaptor) {
  Attribute attr = adaptor.getOffset();
  if (!attr)
    return nullptr;
  if (llvm::APInt value; m_ConstantInt(&value).match(attr) && value.isZero())
    return getBase();
  return nullptr;
}

LogicalResult PtrAddOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, PropertyRef properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // Get the base pointer and offset types.
  Type baseType = operands[0].getType();
  Type offsetType = operands[1].getType();

  auto offTy = dyn_cast<ShapedType>(offsetType);
  if (!offTy) {
    // If the offset isn't shaped, the result is always the base type.
    inferredReturnTypes.push_back(baseType);
    return success();
  }
  auto baseTy = dyn_cast<ShapedType>(baseType);
  if (!baseTy) {
    // Base isn't shaped, but offset is, use the ShapedType from offset with the
    // base pointer as element type.
    inferredReturnTypes.push_back(offTy.clone(baseType));
    return success();
  }

  // Both are shaped, their shape must match.
  if (offTy.getShape() != baseTy.getShape()) {
    if (location)
      mlir::emitError(*location) << "shapes of base and offset must match";
    return failure();
  }

  // Make sure they are the same kind of shaped type.
  if (baseType.getTypeID() != offsetType.getTypeID()) {
    if (location)
      mlir::emitError(*location) << "the shaped containers type must match";
    return failure();
  }
  inferredReturnTypes.push_back(baseType);
  return success();
}

//===----------------------------------------------------------------------===//
// PtrDiffOp
//===----------------------------------------------------------------------===//

LogicalResult PtrDiffOp::verify() {
  // If the operands are not shaped early exit.
  if (!isa<ShapedType>(getLhs().getType()))
    return success();

  // Just check the container type matches, `SameOperandsAndResultShape` handles
  // the actual shape.
  if (getResult().getType().getTypeID() != getLhs().getType().getTypeID()) {
    return emitError() << "expected the result to have the same container "
                          "type as the operands when operands are shaped";
  }

  return success();
}

ptr::PtrType PtrDiffOp::getPtrType() {
  Type lhsType = getLhs().getType();
  if (auto shapedType = dyn_cast<ShapedType>(lhsType))
    return cast<ptr::PtrType>(shapedType.getElementType());
  return cast<ptr::PtrType>(lhsType);
}

Type PtrDiffOp::getIntType() {
  Type resultType = getResult().getType();
  if (auto shapedType = dyn_cast<ShapedType>(resultType))
    return shapedType.getElementType();
  return resultType;
}

//===----------------------------------------------------------------------===//
// ToPtrOp
//===----------------------------------------------------------------------===//

OpFoldResult ToPtrOp::fold(FoldAdaptor adaptor) {
  // Fold the pattern:
  // %val = ptr.from_ptr %p (metadata ...)? : ptr -> type
  // %ptr = ptr.to_ptr %val : type -> ptr
  // To:
  // %ptr -> %p
  Value ptr;
  ToPtrOp toPtr = *this;
  while (toPtr != nullptr) {
    auto fromPtr = toPtr.getPtr().getDefiningOp<FromPtrOp>();
    // Cannot fold if it's not a `from_ptr` op.
    if (!fromPtr)
      return ptr;
    ptr = fromPtr.getPtr();
    // Check for chains of casts.
    toPtr = ptr.getDefiningOp<ToPtrOp>();
  }
  return ptr;
}

LogicalResult ToPtrOp::verify() {
  if (isa<PtrType>(getPtr().getType()))
    return emitError() << "the input value cannot be of type `!ptr.ptr`";
  if (getType().getMemorySpace() != getPtr().getType().getMemorySpace()) {
    return emitError()
           << "expected the input and output to have the same memory space";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// TypeOffsetOp
//===----------------------------------------------------------------------===//

llvm::TypeSize TypeOffsetOp::getTypeSize(std::optional<DataLayout> layout) {
  if (layout)
    return layout->getTypeSize(getElementType());
  DataLayout dl = DataLayout::closest(*this);
  return dl.getTypeSize(getElementType());
}

//===----------------------------------------------------------------------===//
// Pointer API.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Ptr/IR/PtrOpsDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Ptr/IR/PtrOpsAttrs.cpp.inc"

#include "mlir/Dialect/Ptr/IR/PtrOpsEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// FutureType custom assembly helpers.
//===----------------------------------------------------------------------===//

/// Parse an optional `<kind> ':'` prefix for `FutureType`. When no kind
/// keyword is present the kind defaults to `opaque`.
static ParseResult parseFutureKind(AsmParser &parser, FutureKind &kind) {
  StringRef keyword;
  if (succeeded(parser.parseOptionalKeyword(&keyword, {"read", "write"}))) {
    if (parser.parseColon())
      return failure();
    kind = *symbolizeFutureKind(keyword);
    return success();
  }
  kind = FutureKind::opaque;
  return success();
}

/// Print the kind prefix (`read :` / `write :`) for `FutureType`. Nothing is
/// printed for the default `opaque` kind.
static void printFutureKind(AsmPrinter &printer, FutureKind kind) {
  if (kind != FutureKind::opaque)
    printer << stringifyFutureKind(kind) << ": ";
}

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Ptr/IR/PtrOpsTypes.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Ptr/IR/PtrOps.cpp.inc"
