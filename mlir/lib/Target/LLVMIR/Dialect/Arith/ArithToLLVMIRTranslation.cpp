//===- ArithToLLVMIRTranslation.cpp - Translate Arith to LLVM IR ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation between the MLIR Arith dialect and
// LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/Dialect/Arith/ArithToLLVMIRTranslation.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::arith;

//===----------------------------------------------------------------------===//
// AddIOp
//===----------------------------------------------------------------------===//
static LogicalResult convertOp(AddIOp op, llvm::IRBuilderBase &builder,
                               LLVM::ModuleTranslation &moduleTranslation) {
  moduleTranslation.mapValue(op.getResult()) =
      builder.CreateAdd(moduleTranslation.lookupValue(op.getLhs()),
                        moduleTranslation.lookupValue(op.getRhs()), /*Name=*/"",
                        op.hasNoUnsignedWrap(), op.hasNoSignedWrap());
  return success();
}

//===----------------------------------------------------------------------===//
// AddFOp
//===----------------------------------------------------------------------===//

static LogicalResult convertOp(AddFOp op, llvm::IRBuilderBase &builder,
                               LLVM::ModuleTranslation &moduleTranslation) {
  moduleTranslation.mapValue(op.getResult()) =
      builder.CreateFAdd(moduleTranslation.lookupValue(op.getLhs()),
                         moduleTranslation.lookupValue(op.getRhs()));
  return success();
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

static LogicalResult convertOp(ConstantOp op, llvm::IRBuilderBase &builder,
                               LLVM::ModuleTranslation &moduleTranslation) {
  moduleTranslation.mapValue(op.getResult()) = LLVM::detail::getLLVMConstant(
      moduleTranslation.convertType(op.getResult().getType()), op.getValue(),
      op.getLoc(), moduleTranslation);
  return success();
}

//===----------------------------------------------------------------------===//
// MulIOp
//===----------------------------------------------------------------------===//

static LogicalResult convertOp(MulIOp op, llvm::IRBuilderBase &builder,
                               LLVM::ModuleTranslation &moduleTranslation) {
  moduleTranslation.mapValue(op.getResult()) =
      builder.CreateMul(moduleTranslation.lookupValue(op.getLhs()),
                        moduleTranslation.lookupValue(op.getRhs()), /*Name=*/"",
                        op.hasNoUnsignedWrap(), op.hasNoSignedWrap());
  return success();
}

//===----------------------------------------------------------------------===//
// MulFOp
//===----------------------------------------------------------------------===//

static LogicalResult convertOp(MulFOp op, llvm::IRBuilderBase &builder,
                               LLVM::ModuleTranslation &moduleTranslation) {
  moduleTranslation.mapValue(op.getResult()) =
      builder.CreateFMul(moduleTranslation.lookupValue(op.getLhs()),
                         moduleTranslation.lookupValue(op.getRhs()));
  return success();
}

//===----------------------------------------------------------------------===//
// LLVMIRTranslationInterface
//===----------------------------------------------------------------------===//
namespace {
class ArithDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  LogicalResult
  convertOperation(Operation *operation, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const override {
    return llvm::TypeSwitch<Operation *, LogicalResult>(operation)
        .Case([&](AddIOp op) {
          return convertOp(op, builder, moduleTranslation);
        })
        .Case([&](AddFOp op) {
          return convertOp(op, builder, moduleTranslation);
        })
        .Case([&](ConstantOp op) {
          return convertOp(op, builder, moduleTranslation);
        })
        .Case([&](MulIOp op) {
          return convertOp(op, builder, moduleTranslation);
        })
        .Case([&](MulFOp op) {
          return convertOp(op, builder, moduleTranslation);
        })
        .Default([&](Operation *op) {
          return op->emitError("unsupported arith operation: ")
                 << op->getName();
        });
  }
};
} // namespace

void mlir::registerArithDialectTranslation(DialectRegistry &registry) {
  registry.insert<arith::ArithDialect>();
  registry.addExtension(+[](MLIRContext *ctx, arith::ArithDialect *dialect) {
    dialect->addInterfaces<ArithDialectLLVMIRTranslationInterface>();
  });
}

void mlir::registerArithDialectTranslation(MLIRContext &context) {
  DialectRegistry registry;
  registerArithDialectTranslation(registry);
  context.appendDialectRegistry(registry);
}
