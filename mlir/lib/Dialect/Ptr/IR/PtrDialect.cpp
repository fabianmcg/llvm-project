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

unsigned PtrType::getTypeSizeInBits(const DataLayout &dataLayout,
                                    DataLayoutEntryListRef params) const {
  if (std::optional<unsigned> size =
          getPointerDataLayoutEntry(params, *this, PtrDLEntryPos::Size))
    return *size;

  // For other memory spaces, use the size of the pointer to the default memory
  // space.
  return dataLayout.getTypeSizeInBits(
      get(getContext(), getDefaultMemorySpace()));
}

unsigned PtrType::getABIAlignment(const DataLayout &dataLayout,
                                  DataLayoutEntryListRef params) const {
  if (std::optional<unsigned> alignment =
          getPointerDataLayoutEntry(params, *this, PtrDLEntryPos::Abi))
    return *alignment;

  return dataLayout.getTypeABIAlignment(
      get(getContext(), getDefaultMemorySpace()));
}

unsigned PtrType::getPreferredAlignment(const DataLayout &dataLayout,
                                        DataLayoutEntryListRef params) const {
  if (std::optional<unsigned> alignment =
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
// LoadOp type
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

LogicalResult LoadOp::verify() {
  Type valueType = getResult().getType();
  if (auto ms = dyn_cast_or_null<MemorySpaceAttrInterface>(
          getAddr().getType().getMemorySpace());
      ms && !ms.isValidLoad(valueType))
    return emitError("incompatible load type");
  return success();
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

#include "mlir/Dialect/Ptr/IR/PtrOpsDialect.cpp.inc"

#include "mlir/Dialect/Ptr/IR/MemorySpaceInterfaces.cpp.inc"

#include "mlir/Dialect/Ptr/IR/PtrOpsEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Ptr/IR/PtrOpsAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Ptr/IR/PtrOpsTypes.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Ptr/IR/PtrOps.cpp.inc"
