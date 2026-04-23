//===- EmitCTypeInterfaces.td - EmitC Type Interfaces ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the type interfaces used by the EmitC dialect and the
// C++ translator.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_EMITC_IR_EMITCTYPEINTERFACES
#define MLIR_DIALECT_EMITC_IR_EMITCTYPEINTERFACES

include "mlir/IR/OpBase.td"

def CxxTypeInterface : TypeInterface<"CxxTypeInterface"> {
  let cppNamespace = "::mlir::emitc";
  let description = [{
    Interface for types that can be emitted as a C++ type by the EmitC C++
    translator. Any type implementing this interface is automatically accepted
    as a supported EmitC type.
  }];

  let methods = [
    InterfaceMethod<[{
        Returns the C++ type string to emit for this type. Called once per
        type occurrence during translation; implementations may allocate.
      }],
      /*retTy=*/"::std::string",
      /*methodName=*/"getCxxType",
      /*args=*/(ins)
    >,
  ];
}

#endif // MLIR_DIALECT_EMITC_IR_EMITCTYPEINTERFACES
