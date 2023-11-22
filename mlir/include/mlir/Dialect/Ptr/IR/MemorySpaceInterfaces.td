//===-- MemorySpaceInterfaces.td - Memory space interfaces ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines memory space attribute interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef PTR_MEMORYSPACEINTERFACES
#define PTR_MEMORYSPACEINTERFACES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Memory space attribute interface.
//===----------------------------------------------------------------------===//

def MemorySpaceAttrInterface : AttrInterface<"MemorySpaceAttrInterface"> {
  let description = [{
    This interface defines a common API for interacting with the memory model of
    a memory space and the operations in the pointer dialect, giving proper
    semantical meaning to the ops.

    Furthermore, this interface allows concepts such as read-only memory to be
    adequately modeled and enforced.
  }];
  let cppNamespace = "::mlir::ptr";
  let methods = [
    InterfaceMethod<
      /*desc=*/        [{
        Returns the dialect owning the memory model.
      }],
      /*returnType=*/  "Dialect*",
      /*methodName=*/  "getModelOwner",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{ return nullptr; }]
    >,
    InterfaceMethod<
      /*desc=*/        [{
        Returns the memory space as an integer, or 0 if using the default model.
      }],
      /*returnType=*/  "unsigned",
      /*methodName=*/  "getAddressSpace",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{ return 0; }]
    >,
    InterfaceMethod<
      /*desc=*/        [{
        Returns the default memory space as an attribute, or `nullptr` if using
        the default model.
      }],
      /*returnType=*/  "Attribute",
      /*methodName=*/  "getDefaultMemorySpace",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{ return nullptr; }]
    >,
    InterfaceMethod<
      /*desc=*/        [{
        Returns whether a type is loadable in the memory space. The default model
        assumes all types are loadable.
      }],
      /*returnType=*/  "bool",
      /*methodName=*/  "isLoadableType",
      /*args=*/        (ins "Type":$type),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{ return true; }]
    >,
    InterfaceMethod<
      /*desc=*/        [{
        Returns whether a type is storable in the memory space. The default model
        assumes all types are storable.
      }],
      /*returnType=*/  "bool",
      /*methodName=*/  "isStorableType",
      /*args=*/        (ins "Type":$type),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{ return true; }]
    >,
    InterfaceMethod<
      /*desc=*/        [{
        Verifies whether the atomic information of an operation is compatible with
        the memory space.
        The default model assumes the op is compatible
      }],
      /*returnType=*/  "LogicalResult",
      /*methodName=*/  "verifyCompatibleAtomicOp",
      /*args=*/        (ins "ptr::AtomicOpInfo":$atomicInfo,
                            "ArrayRef<ptr::AtomicOrdering>":$unsupportedOrdering),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{ return success(); }]
    >,
    InterfaceMethod<
      /*desc=*/        [{
        Verifies whether an `atomicrmw` op is semantically correct according to
        the memory space.
        The default model assumes the op is compatible.
      }],
      /*returnType=*/  "LogicalResult",
      /*methodName=*/  "verifyAtomicRMW",
      /*args=*/        (ins "ptr::AtomicOpInfo":$atomicInfo,
                            "ptr::AtomicBinOp":$binOp),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{ return success(); }]
    >,
    InterfaceMethod<
      /*desc=*/        [{
        Verifies whether a `cmpxchg` op is semantically correct according to the
        memory space.
        The default model assumes the op is compatible.
      }],
      /*returnType=*/  "LogicalResult",
      /*methodName=*/  "verifyAtomicAtomicCmpXchg",
      /*args=*/        (ins "ptr::AtomicOpInfo":$atomicInfo,
                            "ptr::AtomicOrdering":$failureOrdering),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{ return success(); }]
    >,
    InterfaceMethod<
      /*desc=*/        [{
        Verifies whether the target and source types are compatible with the
        `addrspacecast` op in the memory space.
        Both types are expected to be vectors of rank 1, or scalars of `ptr`
        type.
      }],
      /*returnType=*/  "LogicalResult",
      /*methodName=*/  "verifyPtrCast",
      /*args=*/        (ins "Operation*":$op, "Type":$tgt, "Type":$src),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{ return verifyPtrCastDefaultImpl(op, tgt, src); }]
    >,
    InterfaceMethod<
      /*desc=*/        [{
        Verifies whether the types are compatible with the `ptrtoint` and
        `inttoptr` ops in the memory space.
        The first type is expected to be nteger-like, while the second must be a
        ptr-like type.
      }],
      /*returnType=*/  "LogicalResult",
      /*methodName=*/  "verifyIntCastTypes",
      /*args=*/        (ins "Operation*":$op, "Type":$intLikeTy, "Type":$ptrLikeTy),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{ return verifyIntCastTypesDefaultImpl(op, intLikeTy, ptrLikeTy); }]
    >,
    InterfaceMethod<
      /*desc=*/        [{
         Remove blocking issues of the store op for the`PromotableMemOpInterface`
         interface. For more information see `PromotableMemOpInterface` in
         `Interfaces/MemorySlotInterfaces`.
      }],
      /*returnType=*/  "DeletionKind",
      /*methodName=*/  "removeStoreBlockingUses",
      /*args=*/        (ins "Operation*":$storeOp,
                            "Value":$value,
                            "const MemorySlot &":$slot,
                            "const SmallPtrSetImpl<OpOperand *> &":$blockingUses,
                            "RewriterBase &":$rewriter,
                            "Value":$reachingDefinition),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{ return removeStoreBlockingUsesDefaultImpl(); }]
    >,
  ];
}

def MemorySpaceOpInterface : OpInterface<"MemorySpaceOpInterface"> {
  let description = [{
    An interface for operations with a memory space controlling their semantics.
  }];

  let cppNamespace = "::mlir::ptr";

  let methods = [
    InterfaceMethod<
      /*desc=*/        "Returns the memory space interface controlling the op.",
      /*returnType=*/  "MemorySpace",
      /*methodName=*/  "getMemorySpace",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{}]
      >,
  ];
}
#endif // PTR_MEMORYSPACEINTERFACES
