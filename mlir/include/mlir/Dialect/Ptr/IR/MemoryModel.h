//===-- MemoryModel.h - ptr dialect memory model  ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ptr's dialect memory model class and related
// interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTR_IR_MEMORYMODEL_H
#define MLIR_DIALECT_PTR_IR_MEMORYMODEL_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
class Operation;
namespace ptr {
/// Memory operation validity.
enum class MemOpValidity : uint8_t {
  Valid = 0, /// The operation is valid.
  InvalidType =
      1, /// The type is not compatible with the operation and memory space.
  InvalidAtomicOrdering = 2, /// The atomic ordering is incompatible with the
                             /// operation and memory space.
  InvalidAlignment = 4,      /// The provided alignment is invalid.
};

inline MemOpValidity operator&(MemOpValidity lhs, MemOpValidity rhs) {
  return static_cast<MemOpValidity>(static_cast<uint8_t>(lhs) &
                                    static_cast<uint8_t>(rhs));
}
inline MemOpValidity operator|(MemOpValidity lhs, MemOpValidity rhs) {
  return static_cast<MemOpValidity>(static_cast<uint8_t>(lhs) |
                                    static_cast<uint8_t>(rhs));
}

/// Checks if a specific cast validity flag is set.
inline bool isMemValidityFlagSet(MemOpValidity value, MemOpValidity flag) {
  return (value & flag) == flag;
}

/// Ptr cast operations validity.
enum class CastValidity : uint8_t {
  Valid = 0,                 /// Valid cast operation.
  InvalidSourceType = 1,     /// Invalid source type.
  InvalidTargetType = 2,     /// Invalid target type.
  InvalidVectorRank = 4,     /// One of the operands has an invalid vector rank.
  InvalidScalableVector = 8, /// One of the operands is a scalable vector.
  IncompatibleShapes = 16,   /// The types have incompatible shapes.
};

inline CastValidity operator&(CastValidity lhs, CastValidity rhs) {
  return static_cast<CastValidity>(static_cast<uint8_t>(lhs) &
                                   static_cast<uint8_t>(rhs));
}
inline CastValidity operator|(CastValidity lhs, CastValidity rhs) {
  return static_cast<CastValidity>(static_cast<uint8_t>(lhs) |
                                   static_cast<uint8_t>(rhs));
}

/// Checks if a specific cast validity flag is set.
inline bool isCastFlagSet(CastValidity value, CastValidity flag) {
  return (value & flag) == flag;
}

/// This method checks if it's valid to perform an `addrspacecast` op in the
/// memory space.
/// Compatible types are:
/// Vectors of rank 1, or scalars of `ptr` type.
CastValidity isValidAddrSpaceCastImpl(Type tgt, Type src);

/// This method checks if it's valid to perform a `ptrtoint` or `inttoptr` op in
/// the memory space. `CastValidity::InvalidSourceType` always refers to the
/// 'ptr-like' type and `CastValidity::InvalidTargetType` always refers to the
/// `int-like` type.
/// Compatible types are:
/// IntLikeTy: Vectors of rank 1, or scalars of integer types or `index` type.
/// PtrLikeTy: Vectors of rank 1, or scalars of `ptr` type.
CastValidity isValidPtrIntCastImpl(Type intLikeTy, Type ptrLikeTy);

enum class AtomicBinOp : uint64_t;
enum class AtomicOrdering : uint64_t;
} // namespace ptr
} // namespace mlir

#include "mlir/Dialect/Ptr/IR/MemorySpaceAttrInterfaces.h.inc"

namespace mlir {
namespace ptr {
/// This class wraps the `MemorySpaceAttrInterface` interface, providing a safe
/// mechanism to specify the default behavior assumed by the ptr dialect.
class MemoryModel {
public:
  MemoryModel() = default;
  MemoryModel(std::nullptr_t) {}
  MemoryModel(MemorySpaceAttrInterface memorySpace)
      : memorySpaceAttr(memorySpace), memorySpace(memorySpace) {}
  MemoryModel(Attribute memorySpace)
      : memorySpaceAttr(memorySpace),
        memorySpace(dyn_cast_or_null<MemorySpaceAttrInterface>(memorySpace)) {}

  operator Attribute() const { return memorySpaceAttr; }
  operator MemorySpaceAttrInterface() const { return memorySpace; }
  bool operator==(const MemoryModel &memSpace) const {
    return memSpace.memorySpaceAttr == memorySpaceAttr;
  }

  /// Returns the underlying memory space.
  Attribute getUnderlyingSpace() const { return memorySpaceAttr; }

  /// Returns true if the underlying memory space is null.
  bool isDefaultModel() const { return memorySpace == nullptr; }

  /// Returns the memory space as an integer, or 0 if using the default model.
  unsigned getAddressSpace() const {
    if (memorySpace)
      return memorySpace.getAddressSpace();
    if (auto intAttr = llvm::dyn_cast_or_null<IntegerAttr>(memorySpaceAttr))
      return intAttr.getInt();
    return 0;
  }

  /// Returns the default memory space as an attribute, or nullptr if using the
  /// default model.
  Attribute getDefaultMemorySpace() const {
    return memorySpace ? memorySpace.getDefaultMemorySpace() : nullptr;
  }

  /// This method checks if it's valid to load a value from the memory space
  /// with a specific type, alignment, and atomic ordering. The default model
  /// assumes all values are loadable.
  MemOpValidity isValidLoad(Type type, AtomicOrdering ordering,
                            IntegerAttr alignment) const {
    return memorySpace ? memorySpace.isValidLoad(type, ordering, alignment)
                       : MemOpValidity::Valid;
  }

  /// This method checks if it's valid to store a value in the memory space with
  /// a specific type, alignment, and atomic ordering. The default model assumes
  /// all values are loadable.
  MemOpValidity isValidStore(Type type, AtomicOrdering ordering,
                             IntegerAttr alignment) const {
    return memorySpace ? memorySpace.isValidStore(type, ordering, alignment)
                       : MemOpValidity::Valid;
  }

  /// This method checks if it's valid to perform an atomic operation in the
  /// memory space with a specific type, alignment, and atomic ordering.
  MemOpValidity isValidAtomicOp(AtomicBinOp op, Type type,
                                AtomicOrdering ordering,
                                IntegerAttr alignment) const {
    return memorySpace
               ? memorySpace.isValidAtomicOp(op, type, ordering, alignment)
               : MemOpValidity::Valid;
  }

  /// This method checks if it's valid to perform an atomic operation in the
  /// memory space with a specific type, alignment, and atomic ordering.
  MemOpValidity isValidAtomicXchg(Type type, AtomicOrdering successOrdering,
                                  AtomicOrdering failureOrdering,
                                  IntegerAttr alignment) const {
    return memorySpace ? memorySpace.isValidAtomicXchg(
                             type, successOrdering, failureOrdering, alignment)
                       : MemOpValidity::Valid;
  }

  /// This method checks if it's valid to perform an `addrspacecast` op in the
  /// memory space.
  CastValidity isValidAddrSpaceCast(Type tgt, Type src) const {
    return memorySpace ? memorySpace.isValidAddrSpaceCast(tgt, src)
                       : isValidAddrSpaceCastImpl(tgt, src);
  }

  /// This method checks if it's valid to perform a `ptrtoint` or `inttoptr` op
  /// in the memory space. `CastValidity::InvalidSourceType` always refers to
  /// the 'ptr-like' type and `CastValidity::InvalidTargetType` always refers to
  /// the `int-like` type.
  CastValidity isValidPtrIntCast(Type intLikeTy, Type ptrLikeTy) const {
    return memorySpace ? memorySpace.isValidPtrIntCast(intLikeTy, ptrLikeTy)
                       : isValidPtrIntCastImpl(intLikeTy, ptrLikeTy);
  }

protected:
  /// Underlying memory space.
  Attribute memorySpaceAttr{};
  /// Memory space.
  MemorySpaceAttrInterface memorySpace{};
};
} // namespace ptr
} // namespace mlir

#include "mlir/Dialect/Ptr/IR/MemorySpaceInterfaces.h.inc"

#endif // MLIR_DIALECT_PTR_IR_MEMORYMODEL_H
