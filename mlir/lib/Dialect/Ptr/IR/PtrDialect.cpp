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
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"

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
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/Ptr/IR/PtrOpsTypes.cpp.inc"
      >();
}

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

LogicalResult mlir::ptr::verifyPtrCastDefaultImpl(Operation *op, Type tgt,
                                                  Type src) {
  std::pair<Type, int64_t> tgtInfo = getVecOrScalarInfo(tgt);
  std::pair<Type, int64_t> srcInfo = getVecOrScalarInfo(src);
  if (!isa<PtrType>(tgtInfo.first) || !isa<PtrType>(srcInfo.first))
    return op->emitError() << "invalid ptr-like operand";
  // Check shape validity.
  if (tgtInfo.second == -1 || srcInfo.second == -1)
    return op->emitError() << "vectors of rank != 1 are not supported";
  if (tgtInfo.second == -2 || srcInfo.second == -2)
    return op->emitError()
           << "vectors with scalable dimensions are not supported";
  if (tgtInfo.second != srcInfo.second)
    return op->emitError() << "incompatible operand shapes";
  return success();
}

LogicalResult mlir::ptr::verifyIntCastTypesDefaultImpl(Operation *op,
                                                       Type intLikeTy,
                                                       Type ptrLikeTy) {
  // Check int-like type.
  std::pair<Type, int64_t> intInfo = getVecOrScalarInfo(intLikeTy);
  if (!intInfo.first.isSignlessIntOrIndex())
    return op->emitError() << "invalid int-like type";
  // Check ptr-like type.
  std::pair<Type, int64_t> ptrInfo = getVecOrScalarInfo(ptrLikeTy);
  if (!isa<PtrType>(ptrInfo.first))
    return op->emitError() << "invalid ptr-like type";
  // Check shape validity.
  if (intInfo.second == -1 || ptrInfo.second == -1)
    return op->emitError() << "vectors of rank != 1 are not supported";
  if (intInfo.second == -2 || ptrInfo.second == -2)
    return op->emitError()
           << "vectors with scalable dimensions are not supported";
  if (intInfo.second != ptrInfo.second)
    return op->emitError() << "incompatible operand shapes";
  return success();
}

DeletionKind mlir::ptr::removeStoreBlockingUsesDefaultImpl() {
  return DeletionKind::Delete;
}

//===----------------------------------------------------------------------===//
// Pointer type
//===----------------------------------------------------------------------===//

constexpr const static unsigned kDefaultPointerSizeBits = 64;
constexpr const static unsigned kBitsInByte = 8;
constexpr const static unsigned kDefaultPointerAlignment = 8;

int64_t PtrType::getAddressSpace() const {
  if (auto intAttr = llvm::dyn_cast_or_null<IntegerAttr>(getMemorySpace()))
    return intAttr.getInt();
  else if (auto ms = llvm::dyn_cast_or_null<MemorySpaceAttrInterface>(
               getMemorySpace()))
    return ms.getAddressSpace();
  return 0;
}

Dialect &PtrType::getSharedDialect() const {
  if (auto memSpace =
          llvm::dyn_cast_or_null<MemorySpaceAttrInterface>(getMemorySpace());
      memSpace && memSpace.getModelOwner())
    return *memSpace.getModelOwner();
  return getDialect();
}

Attribute PtrType::getDefaultMemorySpace() const {
  if (auto ms =
          llvm::dyn_cast_or_null<MemorySpaceAttrInterface>(getMemorySpace()))
    return ms.getDefaultMemorySpace();
  return nullptr;
}

std::optional<unsigned> mlir::ptr::extractPointerSpecValue(Attribute attr,
                                                           PtrDLEntryPos pos) {
  auto spec = llvm::cast<DenseIntElementsAttr>(attr);
  auto idx = static_cast<unsigned>(pos);
  if (idx >= spec.size())
    return std::nullopt;
  return spec.getValues<unsigned>()[idx];
}

/// Returns the part of the data layout entry that corresponds to `pos` for the
/// given `type` by interpreting the list of entries `params`. For the pointer
/// type in the default address space, returns the default value if the entries
/// do not provide a custom one, for other address spaces returns std::nullopt.
static std::optional<unsigned>
getPointerDataLayoutEntry(DataLayoutEntryListRef params, PtrType type,
                          PtrDLEntryPos pos) {
  // First, look for the entry for the pointer in the current address space.
  Attribute currentEntry;
  for (DataLayoutEntryInterface entry : params) {
    if (!entry.isTypeEntry())
      continue;
    if (llvm::cast<PtrType>(entry.getKey().get<Type>()).getMemorySpace() ==
        type.getMemorySpace()) {
      currentEntry = entry.getValue();
      break;
    }
  }
  if (currentEntry) {
    return *extractPointerSpecValue(currentEntry, pos) /
           (pos == PtrDLEntryPos::Size ? 1 : kBitsInByte);
  }

  // If not found, and this is the pointer to the default memory space, assume
  // 64-bit pointers.
  if (type.getAddressSpace() == 0) {
    return pos == PtrDLEntryPos::Size ? kDefaultPointerSizeBits
                                      : kDefaultPointerAlignment;
  }

  return std::nullopt;
}

llvm::TypeSize PtrType::getTypeSizeInBits(const DataLayout &dataLayout,
                                          DataLayoutEntryListRef params) const {
  if (std::optional<uint64_t> size =
          getPointerDataLayoutEntry(params, *this, PtrDLEntryPos::Size))
    return llvm::TypeSize::getFixed(*size);

  // For other memory spaces, use the size of the pointer to the default memory
  // space.
  return dataLayout.getTypeSizeInBits(
      get(getContext(), getDefaultMemorySpace()));
}

uint64_t PtrType::getABIAlignment(const DataLayout &dataLayout,
                                  DataLayoutEntryListRef params) const {
  if (std::optional<uint64_t> alignment =
          getPointerDataLayoutEntry(params, *this, PtrDLEntryPos::Abi))
    return *alignment;

  return dataLayout.getTypeABIAlignment(
      get(getContext(), getDefaultMemorySpace()));
}

uint64_t PtrType::getPreferredAlignment(const DataLayout &dataLayout,
                                        DataLayoutEntryListRef params) const {
  if (std::optional<uint64_t> alignment =
          getPointerDataLayoutEntry(params, *this, PtrDLEntryPos::Preferred))
    return *alignment;

  return dataLayout.getTypePreferredAlignment(
      get(getContext(), getDefaultMemorySpace()));
}

bool PtrType::areCompatible(DataLayoutEntryListRef oldLayout,
                            DataLayoutEntryListRef newLayout) const {
  for (DataLayoutEntryInterface newEntry : newLayout) {
    if (!newEntry.isTypeEntry())
      continue;
    unsigned size = kDefaultPointerSizeBits;
    unsigned abi = kDefaultPointerAlignment;
    auto newType = llvm::cast<PtrType>(newEntry.getKey().get<Type>());
    const auto *it =
        llvm::find_if(oldLayout, [&](DataLayoutEntryInterface entry) {
          if (auto type = llvm::dyn_cast_if_present<Type>(entry.getKey())) {
            return llvm::cast<PtrType>(type).getMemorySpace() ==
                   newType.getMemorySpace();
          }
          return false;
        });
    if (it == oldLayout.end()) {
      llvm::find_if(oldLayout, [&](DataLayoutEntryInterface entry) {
        if (auto type = llvm::dyn_cast_if_present<Type>(entry.getKey())) {
          return llvm::cast<PtrType>(type).getAddressSpace() == 0;
        }
        return false;
      });
    }
    if (it != oldLayout.end()) {
      size = *extractPointerSpecValue(*it, PtrDLEntryPos::Size);
      abi = *extractPointerSpecValue(*it, PtrDLEntryPos::Abi);
    }

    Attribute newSpec = llvm::cast<DenseIntElementsAttr>(newEntry.getValue());
    unsigned newSize = *extractPointerSpecValue(newSpec, PtrDLEntryPos::Size);
    unsigned newAbi = *extractPointerSpecValue(newSpec, PtrDLEntryPos::Abi);
    if (size != newSize || abi < newAbi || abi % newAbi != 0)
      return false;
  }
  return true;
}

LogicalResult PtrType::verifyEntries(DataLayoutEntryListRef entries,
                                     Location loc) const {
  for (DataLayoutEntryInterface entry : entries) {
    if (!entry.isTypeEntry())
      continue;
    auto values = llvm::dyn_cast<DenseIntElementsAttr>(entry.getValue());
    if (!values || (values.size() != 3 && values.size() != 4)) {
      return emitError(loc)
             << "expected layout attribute for " << entry.getKey().get<Type>()
             << " to be a dense integer elements attribute with 3 or 4 "
                "elements";
    }
    if (extractPointerSpecValue(values, PtrDLEntryPos::Abi) >
        extractPointerSpecValue(values, PtrDLEntryPos::Preferred)) {
      return emitError(loc) << "preferred alignment is expected to be at least "
                               "as large as ABI alignment";
    }
  }
  return success();
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
} // namespace

//===----------------------------------------------------------------------===//
// AtomicRMWOp
//===----------------------------------------------------------------------===//
LogicalResult AtomicRMWOp::verify() {
  return getMemorySpace().verifyAtomicRMW(getAtomicOpInfo(), getBinOp());
}

MemorySpace AtomicRMWOp::getMemorySpace() {
  return MemorySpace(getPtr().getType().getMemorySpace());
}

//===----------------------------------------------------------------------===//
// AtomicCmpXchgOp
//===----------------------------------------------------------------------===//
LogicalResult AtomicCmpXchgOp::verify() {
  return getMemorySpace().verifyAtomicAtomicCmpXchg(getAtomicOpInfo(),
                                                    getFailureOrdering());
}

MemorySpace AtomicCmpXchgOp::getMemorySpace() {
  return MemorySpace(getPtr().getType().getMemorySpace());
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//
bool ptr::LoadOp::loadsFrom(const MemorySlot &slot) {
  return getAddr() == slot.ptr;
}

bool ptr::LoadOp::storesTo(const MemorySlot &slot) { return false; }

Value ptr::LoadOp::getStored(const MemorySlot &slot, RewriterBase &rewriter) {
  llvm_unreachable("getStored should not be called on LoadOp");
}

bool LoadOp::canUsesBeRemoved(const MemorySlot &slot,
                              const SmallPtrSetImpl<OpOperand *> &blockingUses,
                              SmallVectorImpl<OpOperand *> &newBlockingUses) {
  if (blockingUses.size() != 1)
    return false;
  Value blockingUse = (*blockingUses.begin())->get();
  // If the blocking use is the slot ptr itself, there will be enough
  // context to reconstruct the result of the load at removal time, so it can
  // be removed (provided it loads the exact stored value and is not
  // volatile).
  return blockingUse == slot.ptr && getAddr() == slot.ptr &&
         getResult().getType() == slot.elemType && !getVolatile_();
}

DeletionKind
LoadOp::removeBlockingUses(const MemorySlot &slot,
                           const SmallPtrSetImpl<OpOperand *> &blockingUses,
                           RewriterBase &rewriter, Value reachingDefinition) {
  // `canUsesBeRemoved` checked this blocking use must be the loaded slot
  // pointer.
  rewriter.replaceAllUsesWith(getResult(), reachingDefinition);
  return DeletionKind::Delete;
}

LogicalResult
LoadOp::ensureOnlySafeAccesses(const MemorySlot &slot,
                               SmallVectorImpl<MemorySlot> &mustBeSafelyUsed) {
  return success(getAddr() != slot.ptr || getType() == slot.elemType);
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

MemorySpace LoadOp::getMemorySpace() {
  return MemorySpace(getAddr().getType().getMemorySpace());
}

LogicalResult LoadOp::verify() {
  PtrType valueType = getAddr().getType();
  MemorySpace ms(valueType.getMemorySpace());
  if (!ms.isLoadableType(valueType))
    return emitError("incompatible load type");
  return ms.verifyCompatibleAtomicOp(
      getAtomicOpInfo(), {AtomicOrdering::release, AtomicOrdering::acq_rel});
}

void LoadOp::build(OpBuilder &builder, OperationState &state, Type type,
                   Value addr, unsigned alignment, bool isVolatile,
                   bool isNonTemporal, AtomicOrdering ordering,
                   StringRef syncscope) {
  build(builder, state, type, addr,
        alignment ? builder.getI64IntegerAttr(alignment) : nullptr, isVolatile,
        isNonTemporal, ordering,
        syncscope.empty() ? nullptr : builder.getStringAttr(syncscope));
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//
bool StoreOp::canUsesBeRemoved(const MemorySlot &slot,
                               const SmallPtrSetImpl<OpOperand *> &blockingUses,
                               SmallVectorImpl<OpOperand *> &newBlockingUses) {
  if (blockingUses.size() != 1)
    return false;
  Value blockingUse = (*blockingUses.begin())->get();
  // If the blocking use is the slot ptr itself, dropping the store is
  // fine, provided we are currently promoting its target value. Don't allow a
  // store OF the slot pointer, only INTO the slot pointer.
  return blockingUse == slot.ptr && getAddr() == slot.ptr &&
         getValue() != slot.ptr && getValue().getType() == slot.elemType &&
         !getVolatile_();
}

DeletionKind
StoreOp::removeBlockingUses(const MemorySlot &slot,
                            const SmallPtrSetImpl<OpOperand *> &blockingUses,
                            RewriterBase &rewriter, Value reachingDefinition) {
  return getMemorySpace().removeStoreBlockingUses(
      *this, getValue(), slot, blockingUses, rewriter, reachingDefinition);
}

LogicalResult
StoreOp::ensureOnlySafeAccesses(const MemorySlot &slot,
                                SmallVectorImpl<MemorySlot> &mustBeSafelyUsed) {
  return success(getAddr() != slot.ptr ||
                 getValue().getType() == slot.elemType);
}

bool StoreOp::loadsFrom(const MemorySlot &slot) { return false; }

bool StoreOp::storesTo(const MemorySlot &slot) { return getAddr() == slot.ptr; }

Value StoreOp::getStored(const MemorySlot &slot, RewriterBase &rewriter) {
  return getValue();
}

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

MemorySpace StoreOp::getMemorySpace() {
  return MemorySpace(getAddr().getType().getMemorySpace());
}

LogicalResult StoreOp::verify() {
  PtrType valueType = getAddr().getType();
  MemorySpace ms(valueType.getMemorySpace());
  if (!ms.isStorableType(valueType))
    return emitError("incompatible store type");
  return ms.verifyCompatibleAtomicOp(
      getAtomicOpInfo(), {AtomicOrdering::acquire, AtomicOrdering::acq_rel});
}

void StoreOp::build(OpBuilder &builder, OperationState &state, Value value,
                    Value addr, unsigned alignment, bool isVolatile,
                    bool isNonTemporal, AtomicOrdering ordering,
                    StringRef syncscope) {
  build(builder, state, value, addr,
        alignment ? builder.getI64IntegerAttr(alignment) : nullptr, isVolatile,
        isNonTemporal, ordering,
        syncscope.empty() ? nullptr : builder.getStringAttr(syncscope));
}

//===----------------------------------------------------------------------===//
// AddrSpaceCastOp
//===----------------------------------------------------------------------===//
LogicalResult AddrSpaceCastOp::verify() {
  return getMemorySpace().verifyPtrCast(*this, getRes().getType(),
                                        getArg().getType());
}

MemorySpace AddrSpaceCastOp::getMemorySpace() {
  if (auto ptrTy = getUnderlyingPtrType(getArg().getType()))
    return MemorySpace(ptrTy.getMemorySpace());
  return MemorySpace();
}

static bool forwardToUsers(Operation *op,
                           SmallVectorImpl<OpOperand *> &newBlockingUses) {
  for (Value result : op->getResults())
    for (OpOperand &use : result.getUses())
      newBlockingUses.push_back(&use);
  return true;
}

bool AddrSpaceCastOp::canUsesBeRemoved(
    const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses) {
  return forwardToUsers(*this, newBlockingUses);
}

DeletionKind AddrSpaceCastOp::removeBlockingUses(
    const SmallPtrSetImpl<OpOperand *> &blockingUses, RewriterBase &rewriter) {
  return DeletionKind::Delete;
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

//===----------------------------------------------------------------------===//
// IntToPtrOp
//===----------------------------------------------------------------------===//
LogicalResult IntToPtrOp::verify() {
  return getMemorySpace().verifyIntCastTypes(*this, getArg().getType(),
                                             getRes().getType());
}

MemorySpace IntToPtrOp::getMemorySpace() {
  if (auto ptrTy = getUnderlyingPtrType(getRes().getType()))
    return MemorySpace(ptrTy.getMemorySpace());
  return MemorySpace();
}

//===----------------------------------------------------------------------===//
// PtrToIntOp
//===----------------------------------------------------------------------===//
LogicalResult PtrToIntOp::verify() {
  return getMemorySpace().verifyIntCastTypes(*this, getRes().getType(),
                                             getArg().getType());
}

MemorySpace PtrToIntOp::getMemorySpace() {
  if (auto ptrTy = getUnderlyingPtrType(getArg().getType()))
    return MemorySpace(ptrTy.getMemorySpace());
  return MemorySpace();
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

MemorySpace ConstantOp::getMemorySpace() {
  return MemorySpace(getResult().getType().getMemorySpace());
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
MemorySpace PtrAddOp::getMemorySpace() {
  return MemorySpace(getResult().getType().getMemorySpace());
}

#include "mlir/Dialect/Ptr/IR/PtrOpsDialect.cpp.inc"

#include "mlir/Dialect/Ptr/IR/MemorySpaceInterfaces.cpp.inc"

#include "mlir/Dialect/Ptr/IR/MemorySpaceAttrInterfaces.cpp.inc"

#include "mlir/Dialect/Ptr/IR/PtrOpsEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Ptr/IR/PtrOpsAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Ptr/IR/PtrOpsTypes.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Ptr/IR/PtrOps.cpp.inc"
