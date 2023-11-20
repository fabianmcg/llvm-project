//===- PointerDialect.td - Pointer dialect -----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef PTR_DIALECT
#define PTR_DIALECT

include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Pointer dialect definition.
//===----------------------------------------------------------------------===//

def Ptr_Dialect : Dialect {
    let name = "ptr";
    let summary = "Pointer dialect";
    let cppNamespace = "::mlir::ptr";
    let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Pointer type definitions
//===----------------------------------------------------------------------===//

class Pointer_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Ptr_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def PointerType : Pointer_Type<"Ptr", "ptr", [
    MemRefElementTypeInterface,
    DeclareTypeInterfaceMethods<DataLayoutTypeInterface, [
      "areCompatible", "verifyEntries"]>
  ]> {
  let summary = "Pointer type";
  let description = [{
    The `ptr` type is an opaque pointer type. This type typically represents
    a reference to an object in memory. Pointers are optionally parameterized
    by a memory space.
    Syntax:

    ```mlir
    pointer ::= `ptr` (`<` memory-space `>`)?
    memory-space ::= attribute-value
    ```
    Note: `!ptr.ptr = !ptr.ptr<0 : i64>`
  }];
  let parameters = (ins "Attribute":$memorySpace);
  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
  let builders = [
    TypeBuilder<(ins CArg<"unsigned", "0">:$addressSpace), [{
      return $_get($_ctxt, IntegerAttr::get(IntegerType::get($_ctxt, 64),
                                            addressSpace));
    }]>,
    TypeBuilder<(ins CArg<"Attribute">:$addressSpace), [{
      return $_get($_ctxt, addressSpace != nullptr
                             ? addressSpace
                             : IntegerAttr::get(IntegerType::get($_ctxt, 64),
                                                0));
    }]>
  ];
  let extraClassDeclaration = [{
    /// Returns the memory space as an unsigned number.
    int64_t getAddressSpace();
  }];
}

//===----------------------------------------------------------------------===//
// Base address operation definition.
//===----------------------------------------------------------------------===//

class Pointer_Op<string mnemonic, list<Trait> traits = []> :
        Op<Ptr_Dialect, mnemonic, traits>;

#endif // PTR_DIALECT
