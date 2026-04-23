//===- CppEmitter.h - Helpers to create C++ emitter -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines helpers to emit C++ code using the EmitC dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TARGET_CPP_CPPEMITTER_H
#define MLIR_TARGET_CPP_CPPEMITTER_H

#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
class Operation;
class Value;
class Type;
class Location;
namespace emitc {

/// Internal emitter implementation. External users must go through
/// EmitCContext.
struct CppEmitter;

/// Public facade over the EmitC-to-C++ translator state. Instances are created
/// by the translator and handed to interface implementations
/// (DeclOpInterface, StmtOpInterface, ExprOpInterface, CxxTypeInterface) so
/// that out-of-tree dialects can emit C++ using the same machinery as in-tree
/// EmitC ops.
///
/// This mirrors the role of `LLVM::ModuleTranslation` in the LLVMIR
/// translator.
class EmitCContext {
public:
  explicit EmitCContext(CppEmitter &emitter) : emitter(emitter) {}

  /// Emit an MLIR type as its C++ spelling.
  LogicalResult emitType(Location loc, Type type);

  /// Emit a full operation, optionally followed by a semicolon.
  LogicalResult emitOperation(Operation &op, bool trailingSemicolon);

  /// Emit a value as an operand of the current operation.
  LogicalResult emitOperand(Value value);

  /// Emit the `<type> <name> = ` prefix for the given operation's result(s).
  LogicalResult emitAssignPrefix(Operation &op);

  /// Emit a local variable declaration `<type> <name>;`.
  LogicalResult emitVariableDeclaration(Location loc, Type type,
                                        StringRef name);

  /// Return (and create if needed) the C++ variable name for a Value.
  StringRef getOrCreateName(Value val);

  /// Raw output stream used for code emission.
  raw_indented_ostream &ostream();

  /// Whether an expression is currently being emitted (affects formatting of
  /// sub-expressions).
  bool isEmittingExpression();

private:
  CppEmitter &emitter;
};

/// Translates the given operation to C++ code. The operation or operations in
/// the region of 'op' need almost all be in EmitC dialect. The parameter
/// 'declareVariablesAtTop' enforces that all variables for op results and block
/// arguments are declared at the beginning of the function.
/// If parameter 'fileId' is non-empty, then body of `emitc.file` ops
/// with matching id are emitted.
LogicalResult translateToCpp(Operation *op, raw_ostream &os,
                             bool declareVariablesAtTop = false,
                             StringRef fileId = {});
} // namespace emitc
} // namespace mlir

#endif // MLIR_TARGET_CPP_CPPEMITTER_H
