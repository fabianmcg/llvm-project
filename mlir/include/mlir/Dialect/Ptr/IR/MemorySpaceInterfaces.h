//===-- MemorySpaceInterfaces.h - Memory space interfaces  ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines memory space interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTR_IR_MEMORYSPACEINTERFACES_H
#define MLIR_DIALECT_PTR_IR_MEMORYSPACEINTERFACES_H

#include "mlir/IR/Attributes.h"

namespace mlir {
class Operation;
class RewriterBase;
struct MemorySlot;
enum class DeletionKind : int32_t;
namespace ptr {
enum class AtomicBinOp : uint64_t;
enum class AtomicOrdering : uint64_t;
/// Verifies whether the target and source types are compatible with the
/// `addrspacecast` op in the default memory space.
/// Compatible types are:
/// Vectors of rank 1, or scalars of `ptr` type.
LogicalResult verifyPtrCastDefaultImpl(Operation *op, Type tgt, Type src);
/// Returns whether the target and source types are compatible with the
/// `ptrtoint` and `inttoptr` ops in the memory space.
/// Compatible types are:
/// IntLikeTy: Vectors of rank 1, or scalars of integer types or `index` type.
/// PtrLikeTy: Vectors of rank 1, or scalars of `ptr` type.
LogicalResult verifyIntCastTypesDefaultImpl(Operation *op, Type intLikeTy,
                                            Type ptrLikeTy);
/// Remove blocking issues of the store op for the`PromotableMemOpInterface`
/// interface, the default implementation always deletes the op. For more
/// information see `PromotableMemOpInterface` in
/// `Interfaces/MemorySlotInterfaces`.
DeletionKind removeStoreBlockingUsesDefaultImpl();

/// Utility class for holding the atomic-related information of an operation.
struct AtomicOpInfo {
  AtomicOpInfo(Operation *op, Type valueType, IntegerAttr alignment,
               StringAttr syncScope, AtomicOrdering ordering, bool volatile_)
      : op(op), valueType(valueType), alignment(alignment),
        syncScope(syncScope), ordering(ordering), volatile_(volatile_) {}
  /// Atomic operation.
  Operation *op;
  /// Type of the value being acted on.
  Type valueType;
  /// Alignment of the operation.
  IntegerAttr alignment;
  /// Sync scope of the op.
  StringAttr syncScope;
  /// Atomic ordering of the op.
  AtomicOrdering ordering;
  /// Whether the atomic operation is volatile.
  bool volatile_;
};
} // namespace ptr
} // namespace mlir

#include "mlir/Dialect/Ptr/IR/MemorySpaceAttrInterfaces.h.inc"

namespace mlir {
namespace ptr {
/// This class wraps the `MemorySpaceAttrInterface` interface, providing a safe
/// mechanism to specify the default behavior assumed by the ptr dialect.
class MemorySpace {
public:
  MemorySpace() = default;
  MemorySpace(std::nullptr_t) {}
  MemorySpace(MemorySpaceAttrInterface memorySpace)
      : memorySpace(memorySpace) {}
  MemorySpace(Attribute memorySpace)
      : memorySpace(dyn_cast_or_null<MemorySpaceAttrInterface>(memorySpace)) {}

  /// Returns the underlying memory space.
  MemorySpaceAttrInterface getUnderlyingSpace() const { return memorySpace; }

  /// Returns true if the underlying memory space is null.
  bool isDefaultModel() const { return memorySpace == nullptr; }

  /// Returns the memory space as an integer, or 0 if using the default model.
  unsigned getAddressSpace() const {
    return memorySpace ? memorySpace.getAddressSpace() : 0;
  }

  /// Returns the default memory space as an attribute, or nullptr if using the
  /// default model.
  Attribute getDefaultMemorySpace() const {
    return memorySpace ? memorySpace.getDefaultMemorySpace() : nullptr;
  }

  /// Returns whether a type is loadable in the memory space. The default model
  /// assumes all types are loadable.
  bool isLoadableType(Type type) const {
    return memorySpace ? memorySpace.isLoadableType(type) : true;
  }

  /// Returns whether a type is storable in the memory space. The default model
  /// assumes all types are storable.
  bool isStorableType(Type type) const {
    return memorySpace ? memorySpace.isStorableType(type) : true;
  }

  /// Verifies whether the atomic information of an operation is compatible with
  /// the memory space. The default model assumes the op is compatible.
  LogicalResult verifyCompatibleAtomicOp(
      AtomicOpInfo atomicInfo,
      ArrayRef<AtomicOrdering> unsupportedOrderings) const {
    return memorySpace ? memorySpace.verifyCompatibleAtomicOp(
                             atomicInfo, unsupportedOrderings)
                       : success();
  }

  /// Verifies whether an `atomicrmw` op is semantically correct according to
  /// the memory space. The default model assumes the op is compatible.
  LogicalResult verifyAtomicRMW(AtomicOpInfo atomicInfo,
                                AtomicBinOp binOp) const {
    return memorySpace ? memorySpace.verifyAtomicRMW(atomicInfo, binOp)
                       : success();
  }

  /// Verifies whether a `cmpxchg` op is semantically correct according to the
  /// memory space. The default model assumes the op is compatible.
  LogicalResult
  verifyAtomicAtomicCmpXchg(AtomicOpInfo atomicInfo,
                            AtomicOrdering failureOrdering) const {
    return memorySpace ? memorySpace.verifyAtomicAtomicCmpXchg(atomicInfo,
                                                               failureOrdering)
                       : success();
  }

  /// Verifies whether the target and source types are compatible with the
  /// `addrspacecast` op in the memory space. Both types are expected to be
  /// vectors of rank 1, or scalars of `ptr` type.
  LogicalResult verifyPtrCast(Operation *op, Type tgt, Type src) const {
    return memorySpace ? memorySpace.verifyPtrCast(op, tgt, src)
                       : verifyPtrCastDefaultImpl(op, tgt, src);
  }

  /// Verifies whether the types are compatible with the `ptrtoint` and
  /// `inttoptr` ops in the memory space. The first type is expected to be
  /// integer-like, while the second must be a ptr-like type.
  LogicalResult verifyIntCastTypes(Operation *op, Type intLikeTy,
                                   Type ptrLikeTy) const {
    return memorySpace
               ? memorySpace.verifyIntCastTypes(op, intLikeTy, ptrLikeTy)
               : verifyIntCastTypesDefaultImpl(op, intLikeTy, ptrLikeTy);
  }

  /// Remove blocking issues of the store op for the`PromotableMemOpInterface`
  /// interface. For more information see `PromotableMemOpInterface` in
  /// `Interfaces/MemorySlotInterfaces`.
  DeletionKind
  removeStoreBlockingUses(Operation *storeOp, Value value,
                          const MemorySlot &slot,
                          const SmallPtrSetImpl<OpOperand *> &blockingUses,
                          RewriterBase &rewriter, Value reachingDefinition) {
    return memorySpace
               ? memorySpace.removeStoreBlockingUses(storeOp, value, slot,
                                                     blockingUses, rewriter,
                                                     reachingDefinition)
               : removeStoreBlockingUsesDefaultImpl();
  }

protected:
  /// Underlying memory space.
  MemorySpaceAttrInterface memorySpace{};
};
} // namespace ptr
} // namespace mlir

#include "mlir/Dialect/Ptr/IR/MemorySpaceInterfaces.h.inc"

#endif // MLIR_DIALECT_PTR_IR_MEMORYSPACEINTERFACES_H
