//===- PointerOps.td - Pointer dialect ops -----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://ptr.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef PTR_OPS
#define PTR_OPS

include "mlir/Dialect/Ptr/IR/PtrDialect.td"
include "mlir/Dialect/Ptr/IR/PtrEnums.td"
include "mlir/Dialect/Ptr/IR/MemorySpaceInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// AtomicRMWOp
//===----------------------------------------------------------------------===//
def PtrAtomicRMWType : AnyTypeOf<[AnyFloat, PtrType, AnyInteger]>;
def Ptr_AtomicRMWOp : Pointer_Op<"atomicrmw", [
    TypesMatchWith<"result #0 and operand #1 have the same type",
                    "val", "res", "$_self">,
    DeclareOpInterfaceMethods<MemorySpaceOpInterface>
  ]> {
  let summary = "Atomic read-modify-write operation";
  let description = [{
    The `atomicrmw` operation provides a way to perform a read-modify-write
    sequence that is free from data races. The `bin_op` enumeration specifies
    the modification to perform. The `val` operand represents the new value to
    be applied during the modification. The `ptr` operand represents the
    pointer that the read and write will be performed against.
    The result represents the latest value that was stored.

    Examples:
    ```mlir
      %old = ptr.atomicrmw volatile add %ptr, %val acq_rel `:` !ptr.ptr , i32
    ```
  }];
  let arguments = (ins AtomicBinOp:$bin_op,
                       PtrType:$ptr,
                       PtrAtomicRMWType:$val,
                       AtomicOrdering:$ordering,
                       OptionalAttr<StrAttr>:$syncscope,
                       OptionalAttr<I64Attr>:$alignment,
                       UnitAttr:$volatile_);
  let results = (outs PtrAtomicRMWType:$res);
  let assemblyFormat = [{
    (`volatile` $volatile_^)? $bin_op $ptr `,` $val
    (`syncscope` `(` $syncscope^ `)`)? $ordering attr-dict `:`
    qualified(type($ptr)) `,` type($val)
  }];
  let builders = [
    OpBuilder<(ins "ptr::AtomicBinOp":$binOp, "Value":$ptr, "Value":$val,
      "ptr::AtomicOrdering":$ordering,
      CArg<"StringRef", "StringRef()">:$syncscope,
      CArg<"unsigned", "0">:$alignment, CArg<"bool", "false">:$isVolatile
    )>
  ];
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    AtomicOpInfo getAtomicOpInfo() {
      return AtomicOpInfo(*this, getVal().getType(), getAlignmentAttr(),
        getSyncscopeAttr(), getOrdering(), getVolatile_());
    }
  }];
}

//===----------------------------------------------------------------------===//
// AtomicCmpXchgOp
//===----------------------------------------------------------------------===//
def Ptr_AtomicCmpXchgType : AnyTypeOf<[AnyInteger, PtrType]>;
def Ptr_AtomicCmpXchgOp : Pointer_Op<"cmpxchg", [
    TypesMatchWith<"operand #1 and operand #2 have the same type",
                   "val", "cmp", "$_self">,
    DeclareOpInterfaceMethods<MemorySpaceOpInterface>
  ]> {
  let summary = "Atomic compare exchange operation";
  let description = [{
    The`cmpxchg` instruction is used to atomically modify memory. It loads a
    value in memory and compares it to a given value. If they are equal, it
    tries to store a new value into the memory.

    Examples:
    ```mlir
      %res = ptr.cmpxchg %ptr, %cmp, %val acq_rel acq_rel : !ptr.ptr, i32
    ```
  }];
  let arguments = (ins PtrType:$ptr,
                       Ptr_AtomicCmpXchgType:$cmp,
                       Ptr_AtomicCmpXchgType:$val,
                       AtomicOrdering:$success_ordering,
                       AtomicOrdering:$failure_ordering,
                       OptionalAttr<StrAttr>:$syncscope,
                       OptionalAttr<I64Attr>:$alignment,
                       UnitAttr:$weak,
                       UnitAttr:$volatile_);
  let results = (outs AnyType:$res);
  let assemblyFormat = [{
    (`weak` $weak^)? (`volatile` $volatile_^)? $ptr `,` $cmp `,` $val
    (`syncscope` `(` $syncscope^ `)`)? $success_ordering $failure_ordering
    attr-dict `:` qualified(type($ptr)) `,` type($val) `,` type($res)
  }];
  let builders = [
    OpBuilder<(ins "Value":$ptr, "Value":$cmp, "Value":$val,
      "ptr::AtomicOrdering":$successOrdering,
      "ptr::AtomicOrdering":$failureOrdering,
      CArg<"StringRef", "StringRef()">:$syncscope,
      CArg<"unsigned", "0">:$alignment, CArg<"bool", "false">:$isWeak,
      CArg<"bool", "false">:$isVolatile
    )>
  ];
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    AtomicOpInfo getAtomicOpInfo() {
      return AtomicOpInfo(*this, getVal().getType(), getAlignmentAttr(),
        getSyncscopeAttr(), getSuccessOrdering(), getVolatile_());
    }
  }];
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//
def Ptr_LoadOp : Pointer_Op<"load", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<PromotableMemOpInterface>,
    DeclareOpInterfaceMethods<SafeMemorySlotAccessOpInterface>,
    DeclareOpInterfaceMethods<MemorySpaceOpInterface>
  ]> {
  let summary = "Load operation";
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
  let arguments = (ins PtrType:$addr,
                       OptionalAttr<I64Attr>:$alignment,
                       UnitAttr:$volatile_,
                       UnitAttr:$nontemporal,
                       DefaultValuedAttr<
                         AtomicOrdering, "ptr::AtomicOrdering::not_atomic">:$ordering,
                       OptionalAttr<StrAttr>:$syncscope);
  let results = (outs AnyType:$res);
  let assemblyFormat = [{
    (`volatile` $volatile_^)? $addr
    (`atomic` (`syncscope` `(` $syncscope^ `)`)? $ordering^)?
    attr-dict `:` qualified(type($addr)) `->` type($res)
  }];
  let builders = [
    OpBuilder<(ins "Type":$type, "Value":$addr,
      CArg<"unsigned", "0">:$alignment, CArg<"bool", "false">:$isVolatile,
      CArg<"bool", "false">:$isNonTemporal,
      CArg<"ptr::AtomicOrdering", "ptr::AtomicOrdering::not_atomic">:$ordering,
      CArg<"StringRef", "StringRef()">:$syncscope)>
  ];
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    AtomicOpInfo getAtomicOpInfo() {
      return AtomicOpInfo(*this, getRes().getType(), getAlignmentAttr(),
        getSyncscopeAttr(), getOrdering(), getVolatile_());
    }
  }];
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//
def Ptr_StoreOp : Pointer_Op<"store", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<PromotableMemOpInterface>,
    DeclareOpInterfaceMethods<SafeMemorySlotAccessOpInterface>,
    DeclareOpInterfaceMethods<MemorySpaceOpInterface>
  ]> {
  let summary = "Store operation";
  let description = [{
    The `store` operation is used to write to memory. A store may be marked as
    atomic, volatile, and/or nontemporal, and takes a number of optional
    attributes that specify aliasing information.

    An atomic store only supports a limited set of pointer, integer, and
    floating point types, and requires an explicit alignment.

    Examples:
    ```mlir
    // A volatile store of a float variable.
    ptr.store volatile %val, %ptr : f32, !ptr.ptr

    // A nontemporal store of a float variable.
    ptr.store %val, %ptr {nontemporal} : f32, !ptr.ptr

    // An atomic store of an integer variable.
    ptr.store %val, %ptr atomic monotonic {alignment = 8 : i64}
        : i64, !ptr.ptr
    ```
  }];
  let arguments = (ins AnyType:$value,
                       PtrType:$addr,
                       OptionalAttr<I64Attr>:$alignment,
                       UnitAttr:$volatile_,
                       UnitAttr:$nontemporal,
                       DefaultValuedAttr<
                         AtomicOrdering, "ptr::AtomicOrdering::not_atomic">:$ordering,
                       OptionalAttr<StrAttr>:$syncscope);
  let assemblyFormat = [{
    (`volatile` $volatile_^)? $value `,` $addr
    (`atomic` (`syncscope` `(` $syncscope^ `)`)? $ordering^)?
    attr-dict `:` type($value) `,` qualified(type($addr))
  }];
  let builders = [
    OpBuilder<(ins "Value":$value, "Value":$addr,
      CArg<"unsigned", "0">:$alignment, CArg<"bool", "false">:$isVolatile,
      CArg<"bool", "false">:$isNonTemporal,
      CArg<"ptr::AtomicOrdering", "ptr::AtomicOrdering::not_atomic">:$ordering,
      CArg<"StringRef", "StringRef()">:$syncscope)>
  ];
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    AtomicOpInfo getAtomicOpInfo() {
      return AtomicOpInfo(*this, getValue().getType(), getAlignmentAttr(),
        getSyncscopeAttr(), getOrdering(), getVolatile_());
    }
  }];
}

//===----------------------------------------------------------------------===//
// AddrSpaceCastOp
//===----------------------------------------------------------------------===//
def Ptr_AddrSpaceCastOp : Pointer_Op<"addrspacecast", [
    DeclareOpInterfaceMethods<PromotableOpInterface>,
    DeclareOpInterfaceMethods<MemorySpaceOpInterface>
  ]> {
  let summary = "Address space cast operation";
  let description = [{
    The `addrspacecast` operation casts pointers between memory spaces.

    Example:
    ```mlir
    %ptr = ptr.addrspacecast %addr : !ptr.ptr to !ptr.ptr<1 : i32>
    ```
  }];
  let arguments = (ins AnyType:$arg);
  let results = (outs AnyType:$res);
  let assemblyFormat = "$arg attr-dict `:` type($arg) `to` type($res)";
  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// IntToPtrOp
//===----------------------------------------------------------------------===//
def Ptr_IntToPtrOp : Pointer_Op<"inttoptr", [
    DeclareOpInterfaceMethods<MemorySpaceOpInterface>
  ]> {
  let summary = "Integer to a pointer operation";
  let description = [{
    The `inttoptr` operation casts an int or index value to a pointer.

    Example:
    ```mlir
    %ptr = ptr.inttoptr %int : i32 to !ptr.ptr<1 : i32>
    ```
  }];
  let arguments = (ins AnyType:$arg);
  let results = (outs AnyType:$res);
  let assemblyFormat = "$arg attr-dict `:` type($arg) `to` type($res)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// PtrToIntOp
//===----------------------------------------------------------------------===//
def Ptr_PtrToIntOp : Pointer_Op<"ptrtoint", [
    DeclareOpInterfaceMethods<MemorySpaceOpInterface>
  ]> {
  let summary = "Pointer to an integer operation";
  let description = [{
    The `ptrtoint` operation casts a pointer value to an int or index.

    Example:
    ```mlir
    %int = ptr.ptrtoint %ptr : !ptr.ptr<1 : i32> to i32
    ```
  }];
  let arguments = (ins AnyType:$arg);
  let results = (outs AnyType:$res);
  let assemblyFormat = "$arg attr-dict `:` type($arg) `to` type($res)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//
def Ptr_ConstantOp : Pointer_Op<"constant", [
    ConstantLike, Pure,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    DeclareOpInterfaceMethods<MemorySpaceOpInterface>
  ]> {
  let summary = "Pointer constant operation";
  let description = [{
    The `ptr.constant` operation produces a pointer-typed SSA value equal to
    some index constant.

    Example:

    ```mlir
    %ptr = ptr.constant 0
    %ptr = ptr.constant 1 : !ptr.ptr<3 : i32>
    ```
  }];
  let arguments = (ins IndexAttr:$value);
  let results = (outs PtrType:$result);
  let builders = [
    OpBuilder<(ins "int64_t":$value, CArg<"Attribute", "nullptr">:$addressSpace)>
  ];
  let assemblyFormat = "attr-dict $value custom<PtrType>(type($result))";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// TypeOffsetOp
//===----------------------------------------------------------------------===//
def Ptr_TypeOffsetOp : Pointer_Op<"type_offset", [ConstantLike, Pure]> {
  let summary = "Type offset constant operation";
  let description = [{
    The `type_offset` operation produces an int or index-typed SSA value
    equal to a target-specific constant representing the offset of a single
    element of the given type. The default return type is `index`.
    Example:

    ```mlir
    %0 = ptr.type_offset f32
    %1 = ptr.type_offset memref<12 x f64> : i32
    ```
  }];

  let arguments = (ins TypeAttr:$baseType);
  let results = (outs AnySignlessIntegerOrIndex:$result);
  let builders = [
    OpBuilder<(ins "TypeAttr":$baseType, CArg<"Type", "nullptr">:$resultTy)>
  ];
  let assemblyFormat = [{
     attr-dict $baseType custom<IntType>(type($result))
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// PtrAddOp
//===----------------------------------------------------------------------===//
def Ptr_PtrAddOp : Pointer_Op<"ptradd", [
    Pure, AllTypesMatch<["base", "result"]>,
    DeclareOpInterfaceMethods<MemorySpaceOpInterface>
  ]> {
  let summary = "Pointer-index add operation";
  let description = [{
    The `ptradd` operation adds an `address` and an integer or index to
    produce a new address.

    Example:
    ```mlir
    %addr = ptr.ptradd %addr : !ptr.ptr<3 : i32>, %c10 : i32
    ```
  }];

  let arguments = (ins PtrType:$base, AnySignlessIntegerOrIndex:$offset);
  let results = (outs PtrType:$result);
  let assemblyFormat = [{
    $base custom<PtrType>(type($base)) `,` $offset
    custom<IntType>(type($offset)) attr-dict
  }];
}

#endif // PTR_OPS
