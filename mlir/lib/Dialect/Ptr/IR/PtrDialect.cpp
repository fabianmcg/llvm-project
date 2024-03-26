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

#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::ptr;

//===----------------------------------------------------------------------===//
// Pointer dialect
//===----------------------------------------------------------------------===//

namespace {
/// This class defines the interface for handling inlining for ptr
/// dialect operations.
struct PtrInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// All ptr dialect ops can be inlined.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }
};
} // namespace

void PtrDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Ptr/IR/PtrOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/Ptr/IR/PtrOpsTypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/Ptr/IR/PtrOpsAttrs.cpp.inc"
      >();
  addInterfaces<PtrInlinerInterface>();
}

//===----------------------------------------------------------------------===//
// Pointer API.
//===----------------------------------------------------------------------===//

// Returns the underlying ptr-type or null.
static PtrType getUnderlyingPtrType(Type ty) {
  Type elemTy = ty;
  if (auto vecTy = dyn_cast<VectorType>(ty))
    elemTy = vecTy.getElementType();
  return dyn_cast<PtrType>(elemTy);
}

// Returns a pair containing:
// The underlying type of a vector or the type itself if it's not a vector.
// The number of elements in the vector or an error code if the type is not
// supported.
static std::pair<Type, int64_t> getVecOrScalarInfo(Type ty) {
  if (auto vecTy = dyn_cast<VectorType>(ty)) {
    auto elemTy = vecTy.getElementType();
    // Vectors of rank greater than one or with scalable dimensions are not
    // supported.
    if (vecTy.getRank() != 1)
      return {elemTy, -1};
    else if (vecTy.getScalableDims()[0])
      return {elemTy, -2};
    return {elemTy, vecTy.getShape()[0]};
  }
  // `ty` is a scalar type.
  return {ty, 0};
}

CastValidity mlir::ptr::isValidAddrSpaceCastImpl(Type tgt, Type src) {
  std::pair<Type, int64_t> tgtInfo = getVecOrScalarInfo(tgt);
  std::pair<Type, int64_t> srcInfo = getVecOrScalarInfo(src);
  if (!isa<PtrType>(tgtInfo.first) || !isa<PtrType>(srcInfo.first))
    return CastValidity::InvalidTargetType | CastValidity::InvalidSourceType;
  // Check shape validity.
  if (tgtInfo.second == -1 || srcInfo.second == -1)
    return CastValidity::InvalidVectorRank;
  if (tgtInfo.second == -2 || srcInfo.second == -2)
    return CastValidity::InvalidScalableVector;
  if (tgtInfo.second != srcInfo.second)
    return CastValidity::IncompatibleShapes;
  return CastValidity::Valid;
}

CastValidity mlir::ptr::isValidPtrIntCastImpl(Type intLikeTy, Type ptrLikeTy) {
  // Check int-like type.
  std::pair<Type, int64_t> intInfo = getVecOrScalarInfo(intLikeTy);
  if (!intInfo.first.isSignlessIntOrIndex())
    /// The int-like operand is invalid.
    return CastValidity::InvalidTargetType;
  // Check ptr-like type.
  std::pair<Type, int64_t> ptrInfo = getVecOrScalarInfo(ptrLikeTy);
  if (!isa<PtrType>(ptrInfo.first))
    /// The pointer-like operand is invalid.
    return CastValidity::InvalidSourceType;
  // Check shape validity.
  if (intInfo.second == -1 || ptrInfo.second == -1)
    return CastValidity::InvalidVectorRank;
  if (intInfo.second == -2 || ptrInfo.second == -2)
    return CastValidity::InvalidScalableVector;
  if (intInfo.second != ptrInfo.second)
    return CastValidity::IncompatibleShapes;
  return CastValidity::Valid;
}

//===----------------------------------------------------------------------===//
// Pointer operations.
//===----------------------------------------------------------------------===//

namespace {
ParseResult parsePtrType(OpAsmParser &parser, Type &ty) {
  if (succeeded(parser.parseOptionalColon()) && parser.parseType(ty))
    return parser.emitError(parser.getNameLoc(), "expected a type");
  if (!ty)
    ty = parser.getBuilder().getType<PtrType>();
  return success();
}
void printPtrType(OpAsmPrinter &p, Operation *op, PtrType ty) {
  if (ty.getMemorySpace() != nullptr)
    p << " : " << ty;
}

ParseResult parseIntType(OpAsmParser &parser, Type &ty) {
  if (succeeded(parser.parseOptionalColon()) && parser.parseType(ty))
    return parser.emitError(parser.getNameLoc(), "expected a type");
  if (!ty)
    ty = parser.getBuilder().getIndexType();
  return success();
}
void printIntType(OpAsmPrinter &p, Operation *op, Type ty) {
  if (!ty.isIndex())
    p << " : " << ty;
}

/// Verifies the validity of a memory operation.
LogicalResult verifyMemOpValidity(Operation *op, Type valueType,
                                  MemOpValidity validity, StringRef typeError) {
  if (isMemValidityFlagSet(validity, MemOpValidity::InvalidType |
                                         MemOpValidity::InvalidAtomicOrdering))
    return op->emitError("unsupported type ")
           << valueType << " for atomic access";
  if (isMemValidityFlagSet(validity, MemOpValidity::InvalidType))
    return op->emitError(typeError);
  if (isMemValidityFlagSet(validity, MemOpValidity::InvalidAtomicOrdering))
    return op->emitError("incompatible atomic ordering");
  if (isMemValidityFlagSet(validity, MemOpValidity::InvalidAlignment))
    return op->emitError("incompatible alignment");
  return success();
}

/// Verifies the attributes and the type of atomic memory access operations.
template <typename OpTy>
LogicalResult verifyAtomicMemOp(OpTy memOp, Type valueType,
                                ArrayRef<AtomicOrdering> unsupportedOrderings) {
  if (memOp.getOrdering() != AtomicOrdering::not_atomic) {
    // if (!isTypeCompatibleWithAtomicOp(valueType,
    //                                   /*isPointerTypeAllowed=*/true))
    //   return memOp.emitOpError("unsupported type ")
    //          << valueType << " for atomic access";
    if (llvm::is_contained(unsupportedOrderings, memOp.getOrdering()))
      return memOp.emitOpError("unsupported ordering '")
             << stringifyAtomicOrdering(memOp.getOrdering()) << "'";
    if (!memOp.getAlignment())
      return memOp.emitOpError("expected alignment for atomic access");
    return success();
  }
  if (memOp.getSyncscope())
    return memOp.emitOpError(
        "expected syncscope to be null for non-atomic access");
  return success();
}
} // namespace

//===----------------------------------------------------------------------===//
// AtomicRMWOp
//===----------------------------------------------------------------------===//

void AtomicRMWOp::build(OpBuilder &builder, OperationState &state,
                        AtomicBinOp binOp, Value ptr, Value val,
                        AtomicOrdering ordering, StringRef syncscope,
                        unsigned alignment, bool isVolatile) {
  build(builder, state, val.getType(), binOp, ptr, val, ordering,
        !syncscope.empty() ? builder.getStringAttr(syncscope) : nullptr,
        alignment ? builder.getI64IntegerAttr(alignment) : nullptr, isVolatile,
        /*access_groups=*/nullptr,
        /*alias_scopes=*/nullptr, /*noalias_scopes=*/nullptr,
        /*tbaa=*/nullptr);
}

SmallVector<Value> AtomicRMWOp::getAccessedOperands() { return {getPtr()}; }

MemoryModel AtomicRMWOp::getMemoryModel() {
  return getPtr().getType().getMemoryModel();
}

LogicalResult AtomicRMWOp::verify() {
  Type valueType = getVal().getType();
  if (failed(verifyMemOpValidity(
          *this, valueType,
          getMemoryModel().isValidAtomicOp(getBinOp(), valueType, getOrdering(),
                                           getAlignmentAttr()),
          "type is invalid")))
    return failure();
  if (static_cast<unsigned>(getOrdering()) <
      static_cast<unsigned>(AtomicOrdering::monotonic))
    return emitOpError() << "expected at least '"
                         << stringifyAtomicOrdering(AtomicOrdering::monotonic)
                         << "' ordering";
  return success();
}

//===----------------------------------------------------------------------===//
// AtomicCmpXchgOp
//===----------------------------------------------------------------------===//

void AtomicCmpXchgOp::build(OpBuilder &builder, OperationState &state,
                            Value ptr, Value cmp, Value val,
                            AtomicOrdering successOrdering,
                            AtomicOrdering failureOrdering, StringRef syncscope,
                            unsigned alignment, bool isWeak, bool isVolatile) {
  build(builder, state, val.getType(), builder.getI1Type(), ptr, cmp, val,
        successOrdering, failureOrdering,
        !syncscope.empty() ? builder.getStringAttr(syncscope) : nullptr,
        alignment ? builder.getI64IntegerAttr(alignment) : nullptr, isWeak,
        isVolatile, /*access_groups=*/nullptr,
        /*alias_scopes=*/nullptr, /*noalias_scopes=*/nullptr, /*tbaa=*/nullptr);
}

SmallVector<Value> AtomicCmpXchgOp::getAccessedOperands() { return {getPtr()}; }

MemoryModel AtomicCmpXchgOp::getMemoryModel() {
  return getPtr().getType().getMemoryModel();
}

LogicalResult AtomicCmpXchgOp::verify() {
  Type valueType = getVal().getType();
  if (failed(verifyMemOpValidity(*this, valueType,
                                 getMemoryModel().isValidAtomicXchg(
                                     valueType, getSuccessOrdering(),
                                     getFailureOrdering(), getAlignmentAttr()),
                                 "type is invalid")))
    return failure();
  if (getSuccessOrdering() < AtomicOrdering::monotonic ||
      getFailureOrdering() < AtomicOrdering::monotonic)
    return emitOpError("ordering must be at least 'monotonic'");
  if (getFailureOrdering() == AtomicOrdering::release ||
      getFailureOrdering() == AtomicOrdering::acq_rel)
    return emitOpError("failure ordering cannot be 'release' or 'acq_rel'");
  return success();
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

void LoadOp::build(OpBuilder &builder, OperationState &state, Type type,
                   Value addr, unsigned alignment, bool isVolatile,
                   bool isNonTemporal, bool isInvariant,
                   AtomicOrdering ordering, StringRef syncscope) {
  build(builder, state, type, addr,
        alignment ? builder.getI64IntegerAttr(alignment) : nullptr, isVolatile,
        isNonTemporal, isInvariant, ordering,
        syncscope.empty() ? nullptr : builder.getStringAttr(syncscope),
        /*access_groups=*/nullptr,
        /*alias_scopes=*/nullptr, /*noalias_scopes=*/nullptr,
        /*tbaa=*/nullptr);
}

void LoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), getAddr());
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

MemoryModel LoadOp::getMemoryModel() {
  return getAddr().getType().getMemoryModel();
}

SmallVector<Value> LoadOp::getAccessedOperands() { return {getAddr()}; }

LogicalResult LoadOp::verify() {
  Type valueType = getRes().getType();
  if (failed(
          verifyMemOpValidity(*this, valueType,
                              getMemoryModel().isValidLoad(
                                  valueType, getOrdering(), getAlignmentAttr()),
                              "type is not loadable")))
    return failure();
  return verifyAtomicMemOp(*this, valueType,
                           {AtomicOrdering::release, AtomicOrdering::acq_rel});
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

void StoreOp::build(OpBuilder &builder, OperationState &state, Value value,
                    Value addr, unsigned alignment, bool isVolatile,
                    bool isNonTemporal, AtomicOrdering ordering,
                    StringRef syncscope) {
  build(builder, state, value, addr,
        alignment ? builder.getI64IntegerAttr(alignment) : nullptr, isVolatile,
        isNonTemporal, ordering,
        syncscope.empty() ? nullptr : builder.getStringAttr(syncscope),
        /*access_groups=*/nullptr,
        /*alias_scopes=*/nullptr, /*noalias_scopes=*/nullptr,
        /*tbaa=*/nullptr);
}

SmallVector<Value> StoreOp::getAccessedOperands() { return {getAddr()}; }

void StoreOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), getAddr());
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

MemoryModel StoreOp::getMemoryModel() {
  return getAddr().getType().getMemoryModel();
}

LogicalResult StoreOp::verify() {
  Type valueType = getValue().getType();
  if (failed(
          verifyMemOpValidity(*this, valueType,
                              getMemoryModel().isValidStore(
                                  valueType, getOrdering(), getAlignmentAttr()),
                              "type is not storable")))
    return failure();
  return verifyAtomicMemOp(*this, valueType,
                           {AtomicOrdering::acquire, AtomicOrdering::acq_rel});
}

//===----------------------------------------------------------------------===//
// AddrSpaceCastOp
//===----------------------------------------------------------------------===//

MemoryModel AddrSpaceCastOp::getMemoryModel() {
  if (auto ptrTy = getUnderlyingPtrType(getArg().getType()))
    return ptrTy.getMemoryModel();
  return MemoryModel();
}

OpFoldResult AddrSpaceCastOp::fold(FoldAdaptor adaptor) {
  // addrcast(x : T0, T0) -> x
  if (getArg().getType() == getType())
    return getArg();
  // addrcast(addrcast(x : T0, T1), T0) -> x
  if (auto prev = getArg().getDefiningOp<AddrSpaceCastOp>())
    if (prev.getArg().getType() == getType())
      return prev.getArg();
  return {};
}

LogicalResult AddrSpaceCastOp::verify() {
  CastValidity validity = getMemoryModel().isValidAddrSpaceCast(
      getRes().getType(), getArg().getType());
  if (isCastFlagSet(validity, CastValidity::InvalidSourceType) ||
      isCastFlagSet(validity, CastValidity::InvalidTargetType))
    return emitError() << "invalid ptr-like operand";
  if (isCastFlagSet(validity, CastValidity::InvalidVectorRank))
    return emitError() << "vectors of rank != 1 are not supported";
  if (isCastFlagSet(validity, CastValidity::InvalidScalableVector))
    return emitError() << "vectors with scalable dimensions are not supported";
  if (isCastFlagSet(validity, CastValidity::IncompatibleShapes))
    return emitError() << "incompatible operand shapes";
  return success();
}

//===----------------------------------------------------------------------===//
// IntToPtrOp
//===----------------------------------------------------------------------===//

MemoryModel IntToPtrOp::getMemoryModel() {
  if (auto ptrTy = getUnderlyingPtrType(getRes().getType()))
    return ptrTy.getMemoryModel();
  return MemoryModel();
}

LogicalResult IntToPtrOp::verify() {
  CastValidity validity = getMemoryModel().isValidPtrIntCast(
      getArg().getType(), getRes().getType());
  if (isCastFlagSet(validity, CastValidity::InvalidSourceType))
    return emitError() << "invalid int-like type";
  if (isCastFlagSet(validity, CastValidity::InvalidTargetType))
    return emitError() << "invalid ptr-like type";
  if (isCastFlagSet(validity, CastValidity::InvalidVectorRank))
    return emitError() << "vectors of rank != 1 are not supported";
  if (isCastFlagSet(validity, CastValidity::InvalidScalableVector))
    return emitError() << "vectors with scalable dimensions are not supported";
  if (isCastFlagSet(validity, CastValidity::IncompatibleShapes))
    return emitError() << "incompatible operand shapes";
  return success();
}

//===----------------------------------------------------------------------===//
// PtrToIntOp
//===----------------------------------------------------------------------===//

MemoryModel PtrToIntOp::getMemoryModel() {
  if (auto ptrTy = getUnderlyingPtrType(getArg().getType()))
    return MemoryModel(ptrTy.getMemoryModel());
  return MemoryModel();
}

LogicalResult PtrToIntOp::verify() {
  CastValidity validity = getMemoryModel().isValidPtrIntCast(
      getRes().getType(), getArg().getType());
  if (isCastFlagSet(validity, CastValidity::InvalidSourceType))
    return emitError() << "invalid int-like type";
  if (isCastFlagSet(validity, CastValidity::InvalidTargetType))
    return emitError() << "invalid ptr-like type";
  if (isCastFlagSet(validity, CastValidity::InvalidVectorRank))
    return emitError() << "vectors of rank != 1 are not supported";
  if (isCastFlagSet(validity, CastValidity::InvalidScalableVector))
    return emitError() << "vectors with scalable dimensions are not supported";
  if (isCastFlagSet(validity, CastValidity::IncompatibleShapes))
    return emitError() << "incompatible operand shapes";
  return success();
}

//===----------------------------------------------------------------------===//
// Constant Op
//===----------------------------------------------------------------------===//

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       int64_t value, Attribute addressSpace) {
  build(odsBuilder, odsState, odsBuilder.getType<PtrType>(addressSpace),
        odsBuilder.getIndexAttr(value));
}

void ConstantOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  SmallString<32> buffer;
  llvm::raw_svector_ostream name(buffer);
  name << "ptr" << getValueAttr().getValue();
  setNameFn(getResult(), name.str());
}

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  return adaptor.getValueAttr();
}

MemoryModel ConstantOp::getMemoryModel() {
  return getResult().getType().getMemoryModel();
}

//===----------------------------------------------------------------------===//
// TypeOffset Op
//===----------------------------------------------------------------------===//

OpFoldResult TypeOffsetOp::fold(FoldAdaptor adaptor) {
  return adaptor.getBaseTypeAttr();
}

//===----------------------------------------------------------------------===//
// PtrAdd Op
//===----------------------------------------------------------------------===//

MemoryModel PtrAddOp::getMemoryModel() {
  return getResult().getType().getMemoryModel();
}

//===----------------------------------------------------------------------===//
// Pointer attributes
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// TBAANodeAttr
//===----------------------------------------------------------------------===//

bool TBAANodeAttr::classof(Attribute attr) {
  return llvm::isa<TBAATypeDescriptorAttr, TBAARootAttr>(attr);
}

//===----------------------------------------------------------------------===//
// Pointer interfaces
//===----------------------------------------------------------------------===//

/// Verifies that all elements of `array` are instances of `Attr`.
template <class AttrT>
static LogicalResult isArrayOf(Operation *op, ArrayAttr array) {
  for (Attribute iter : array)
    if (!isa<AttrT>(iter))
      return op->emitOpError("expected op to return array of ")
             << AttrT::getMnemonic() << " attributes";
  return success();
}

//===----------------------------------------------------------------------===//
// AccessGroupOpInterface
//===----------------------------------------------------------------------===//

LogicalResult mlir::ptr::detail::verifyAccessGroupOpInterface(Operation *op) {
  auto iface = cast<AccessGroupOpInterface>(op);
  ArrayAttr accessGroups = iface.getAccessGroupsOrNull();
  if (!accessGroups)
    return success();

  return isArrayOf<AccessGroupAttr>(op, accessGroups);
}

//===----------------------------------------------------------------------===//
// AliasAnalysisOpInterface
//===----------------------------------------------------------------------===//

LogicalResult mlir::ptr::detail::verifyAliasAnalysisOpInterface(Operation *op) {
  auto iface = cast<AliasAnalysisOpInterface>(op);

  if (auto aliasScopes = iface.getAliasScopesOrNull())
    if (failed(isArrayOf<AliasScopeAttr>(op, aliasScopes)))
      return failure();

  if (auto noAliasScopes = iface.getNoAliasScopesOrNull())
    if (failed(isArrayOf<AliasScopeAttr>(op, noAliasScopes)))
      return failure();

  ArrayAttr tags = iface.getTBAATagsOrNull();
  if (!tags)
    return success();

  return isArrayOf<TBAATagAttr>(op, tags);
}

#include "mlir/Dialect/Ptr/IR/PtrOpsDialect.cpp.inc"

#include "mlir/Dialect/Ptr/IR/PtrInterfaces.cpp.inc"

#include "mlir/Dialect/Ptr/IR/MemorySpaceInterfaces.cpp.inc"

#include "mlir/Dialect/Ptr/IR/MemorySpaceAttrInterfaces.cpp.inc"

#include "mlir/Dialect/Ptr/IR/PtrOpsEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Ptr/IR/PtrOpsAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Ptr/IR/PtrOpsTypes.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Ptr/IR/PtrOps.cpp.inc"
