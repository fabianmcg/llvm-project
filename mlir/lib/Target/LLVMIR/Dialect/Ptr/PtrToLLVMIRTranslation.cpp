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
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/MDBuilder.h"

using namespace mlir;
using namespace mlir::ptr;

namespace {
LogicalResult convertLoadOp(LoadOp op, llvm::IRBuilderBase &builder,
                            LLVM::ModuleTranslation &moduleTranslation) {

  auto *inst = builder.CreateLoad(
      moduleTranslation.convertType(op.getResult().getType()),
      moduleTranslation.lookupValue(op.getAddr()), op.getVolatile_());
  moduleTranslation.mapValue(op.getRes()) = inst;

  if (op.getSyncscope().has_value()) {
    llvm::LLVMContext &llvmContext = builder.getContext();
    inst->setSyncScopeID(
        llvmContext.getOrInsertSyncScopeID(*op.getSyncscope()));
  }

  if (op.getAlignment().has_value()) {
    auto align = *op.getAlignment();
    if (align != 0)
      inst->setAlignment(llvm::Align(align));
  }

  if (op.getNontemporal()) {
    llvm::MDNode *metadata = llvm::MDNode::get(
        inst->getContext(), llvm::ConstantAsMetadata::get(builder.getInt32(1)));
    inst->setMetadata(llvm::LLVMContext::MD_nontemporal, metadata);
  }
  return success();
}

class PtrDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  LogicalResult
  convertOperation(Operation *operation, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const override {
    return llvm::TypeSwitch<Operation *, LogicalResult>(operation)
        .Case([&](ptr::LoadOp op) {
          return convertLoadOp(op, builder, moduleTranslation);
        })
        .Default([&](Operation *op) {
          return op->emitError("unsupported Ptr operation: ") << op->getName();
        });
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
