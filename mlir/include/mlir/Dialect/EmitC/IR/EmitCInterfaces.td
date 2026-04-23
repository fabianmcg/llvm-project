//===- EmitCInterfaces.td - EmitC Interfaces ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the interfaces used by EmitC.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_EMITC_IR_EMITCINTERFACES
#define MLIR_DIALECT_EMITC_IR_EMITCINTERFACES

include "mlir/IR/OpBase.td"

def DeclOpInterface : OpInterface<"DeclOpInterface"> {
  let description = [{
    Interface for operations that emit a C++ declaration (struct, class,
    type alias, global variable, etc.) into an `EmitCContext`. Implementations
    are responsible for emitting their own trailing semicolon; the translator
    will not append one.
  }];
  let cppNamespace = "::mlir::emitc";
  let methods = [
    InterfaceMethod<
      "Emit this operation as a C++ declaration.",
      "::mlir::LogicalResult", "emitDecl",
      (ins "::mlir::emitc::EmitCContext &":$ctx)>,
  ];
}

def StmtOpInterface : OpInterface<"StmtOpInterface"> {
  let description = [{
    Interface for operations that emit a C++ statement into an EmitCContext.
  }];
  let cppNamespace = "::mlir::emitc";
  let methods = [
    InterfaceMethod<
      "Emit this operation as a C++ statement.",
      "::mlir::LogicalResult", "emitStmt",
      (ins "::mlir::emitc::EmitCContext &":$ctx)>,
  ];
}

def ExprOpInterface : OpInterface<"ExprOpInterface"> {
  let description = [{
    Interface for operations that emit a C++ expression into an EmitCContext.
    Ops implementing this interface may be inlined into an `emitc.expression`
    body using the precedence returned by `getPrecedence`.
  }];
  let cppNamespace = "::mlir::emitc";
  let methods = [
    InterfaceMethod<
      "Emit this operation as a C++ expression.",
      "::mlir::LogicalResult", "emitExpr",
      (ins "::mlir::emitc::EmitCContext &":$ctx)>,
    InterfaceMethod<
      "Return the C++ operator precedence of the produced expression. "
      "The scale follows cppreference.com: 1 (comma) to 17 (primary). "
      "The default 17 treats the expression as a primary (highest-precedence) "
      "operand, which is safe for any self-contained expression.",
      "int", "getPrecedence", (ins), /*methodBody=*/[{}],
      /*defaultImplementation=*/[{ return 17; }]>,
  ];
}

def CExpressionInterface : OpInterface<"CExpressionInterface"> {
  let description = [{
    Interface to mark operations that can be part of the CExpression.
  }];

  let cppNamespace = "::mlir::emitc";
  let methods = [
    InterfaceMethod<[{
      Check whether operation has side effects that may affect the expression
      evaluation.

      By default operation is marked as having side effects.

      ```c++
      class ConcreteOp ... {
      public:
        bool hasSideEffects() {
          // That way we can override the default implementation.
          return false;
        }
      };
      ```
    }],
      "bool", "hasSideEffects", (ins), /*methodBody=*/[{}],
       /*defaultImplementation=*/[{
        return true;
    }]>,
  ];
}

#endif // MLIR_DIALECT_EMITC_IR_EMITCINTERFACES
