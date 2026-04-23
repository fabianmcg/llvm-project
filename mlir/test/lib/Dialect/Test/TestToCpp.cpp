//===- TestToCpp.cpp - Translate Test dialect through the Cpp emitter -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Registers a translation that runs the Cpp emitter with the test dialect
// available. This exercises the external EmitC interfaces (DeclOpInterface,
// StmtOpInterface, ExprOpInterface, CxxTypeInterface) implemented by test ops
// and types.
//
//===----------------------------------------------------------------------===//

#include "TestDialect.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "mlir/Tools/mlir-translate/Translation.h"

using namespace mlir;

namespace mlir {

void registerTestToCpp() {
  TranslateFromMLIRRegistration registration(
      "test-to-cpp", "translate from test+emitc to cpp",
      [](Operation *op, raw_ostream &output) {
        return emitc::translateToCpp(op, output);
      },
      [](DialectRegistry &registry) {
        registry.insert<cf::ControlFlowDialect, emitc::EmitCDialect,
                        func::FuncDialect, test::TestDialect>();
      });
}

} // namespace mlir
