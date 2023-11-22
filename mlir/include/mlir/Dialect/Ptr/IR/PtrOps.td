//===- PointerOps.td - Pointer dialect ops -----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef PTR_OPS
#define PTR_OPS

include "mlir/Dialect/Ptr/IR/PtrDialect.td"
include "mlir/Dialect/Ptr/IR/PtrEnums.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/EnumAttr.td"


def Ptr_LoadOp : Pointer_Op<"load",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
     DeclareOpInterfaceMethods<PromotableMemOpInterface>,
     DeclareOpInterfaceMethods<SafeMemorySlotAccessOpInterface>]> {
  let arguments = (ins PtrType:$addr,
              OptionalAttr<I64Attr>:$alignment,
              UnitAttr:$volatile_,
              UnitAttr:$nontemporal,
              DefaultValuedAttr<
                AtomicOrdering, "AtomicOrdering::not_atomic">:$ordering,
              OptionalAttr<StrAttr>:$syncscope);
  let results = (outs AnyType:$res);
  string llvmInstName = "Load";
  let description = [{
    The `load` operation is used to read from memory. A load may be marked as
    atomic, volatile, and/or nontemporal, and takes a number of optional
    attributes that specify aliasing information.

    An atomic load only supports a limited set of pointer, integer, and
    floating point types, and requires an explicit alignment.

    Examples:
    ```mlir
    // A volatile load of a float variable.
    %0 = ptr.load volatile %ptr : !ptr.ptr -> f32

    // A nontemporal load of a float variable.
    %0 = ptr.load %ptr {nontemporal} : !ptr.ptr -> f32

    // An atomic load of an integer variable.
    %0 = ptr.load %ptr atomic monotonic {alignment = 8 : i64}
        : !ptr.ptr -> i64
    ```
  }];
  let assemblyFormat = [{
    (`volatile` $volatile_^)? $addr
    (`atomic` (`syncscope` `(` $syncscope^ `)`)? $ordering^)?
    attr-dict `:` qualified(type($addr)) `->` type($res)
  }];
  let builders = [
    OpBuilder<(ins "Type":$type, "Value":$addr,
      CArg<"unsigned", "0">:$alignment, CArg<"bool", "false">:$isVolatile,
      CArg<"bool", "false">:$isNonTemporal,
      CArg<"AtomicOrdering", "AtomicOrdering::not_atomic">:$ordering,
      CArg<"StringRef", "StringRef()">:$syncscope)>
  ];
  let hasVerifier = 1;
}

#endif // PTR_OPS
