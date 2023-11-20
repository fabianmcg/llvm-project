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

Type PtrType::parse(AsmParser &parser) {
  Attribute memorySpace;
  // Parse literal '<'
  if (succeeded(parser.parseOptionalLess())) {
    if (parser.parseAttribute(memorySpace)) {
      parser.emitError(parser.getCurrentLocation(),
                       "failed to parse PtrType parameter 'memorySpace' "
                       "which is to be a `Attribute`");
      return {};
    }
    // Parse literal '>'
    if (parser.parseGreater())
      return {};
  }
  return PtrType::get(parser.getContext(), memorySpace);
}

void PtrType::print(AsmPrinter &printer) const {
  if (auto intAttr = mlir::dyn_cast<IntegerAttr>(getMemorySpace());
      intAttr && intAttr.getType().isInteger(64)) {
    if (intAttr.getInt() != 0)
      printer << "<" << intAttr.getInt() << ">";
  } else {
    printer << "<" << getMemorySpace() << ">";
  }
}

int64_t PtrType::getAddressSpace() {
  if (auto intAttr = llvm::dyn_cast_or_null<IntegerAttr>(getMemorySpace()))
    return intAttr.getInt();
  return 0;
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
  return dataLayout.getTypeSizeInBits(get(getContext()));
}

unsigned PtrType::getABIAlignment(const DataLayout &dataLayout,
                                  DataLayoutEntryListRef params) const {
  if (std::optional<unsigned> alignment =
          getPointerDataLayoutEntry(params, *this, PtrDLEntryPos::Abi))
    return *alignment;

  return dataLayout.getTypeABIAlignment(get(getContext()));
}

unsigned PtrType::getPreferredAlignment(const DataLayout &dataLayout,
                                        DataLayoutEntryListRef params) const {
  if (std::optional<unsigned> alignment =
          getPointerDataLayoutEntry(params, *this, PtrDLEntryPos::Preferred))
    return *alignment;

  return dataLayout.getTypePreferredAlignment(get(getContext()));
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

#include "mlir/Dialect/Ptr/IR/PtrOpsDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Ptr/IR/PtrOpsTypes.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Ptr/IR/PtrOps.cpp.inc"
