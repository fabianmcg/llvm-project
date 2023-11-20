//===- PtrToLLVMIRTranslation.cpp - Translate Ptr dialect to LLVM IR ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR Ptr dialect and LLVM IR.
//
//===----------------------------------------------------------------------===//
#include "mlir/Target/LLVMIR/Dialect/Ptr/PtrToLLVMIRTranslation.h"
#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace {
class PtrDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  LogicalResult
  convertOperation(Operation *operation, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const override {
    return operation->emitError("unsupported Ptr operation: ")
           << operation->getName();
  }
};

} // namespace

void mlir::registerPtrDialectTranslation(DialectRegistry &registry) {
  registry.insert<ptr::PtrDialect>();
  registry.addExtension(+[](MLIRContext *ctx, ptr::PtrDialect *dialect) {
    dialect->addInterfaces<PtrDialectLLVMIRTranslationInterface>();
  });
}

void mlir::registerPtrDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerPtrDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
