//===- ConvertToLLVMIR.cpp - MLIR to LLVM IR conversion -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR LLVM dialect and LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/IR/DebugProgramInstruction.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Format.h"

#include <chrono>

extern llvm::cl::opt<bool> WriteNewDbgInfoFormat;

using namespace mlir;

namespace mlir {
void registerToLLVMIRTranslation() {
  TranslateFromMLIRRegistration registration(
      "mlir-to-llvmir", "Translate MLIR to LLVMIR",
      [](Operation *op, raw_ostream &output) {
        llvm::LLVMContext llvmContext;
        std::chrono::time_point<std::chrono::steady_clock> startTime =
            std::chrono::steady_clock::now();
        auto llvmModule = translateModuleToLLVMIR(op, llvmContext);
        std::chrono::time_point<std::chrono::steady_clock> stopTime =
            std::chrono::steady_clock::now();
        double eTime =
            std::chrono::duration_cast<std::chrono::duration<double>>(stopTime -
                                                                      startTime)
                .count();
        if (!llvmModule)
          return failure();
        llvm::errs() << llvm::format(
            "  Translation to MLIR-LLVMIR: %.8f seconds\n\n", eTime);

        // When printing LLVM IR, we should convert the module to the debug info
        // format that LLVM expects us to print.
        // See https://llvm.org/docs/RemoveDIsDebugInfo.html
        llvm::ScopedDbgInfoFormatSetter formatSetter(*llvmModule,
                                                     WriteNewDbgInfoFormat);
        if (WriteNewDbgInfoFormat)
          llvmModule->removeDebugIntrinsicDeclarations();
        llvmModule->print(output, nullptr);
        return success();
      },
      [](DialectRegistry &registry) {
        registry.insert<DLTIDialect, func::FuncDialect>();
        registerAllToLLVMIRTranslations(registry);
      });
}
} // namespace mlir
