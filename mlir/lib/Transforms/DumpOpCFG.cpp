//===- DumpOpCFG.cpp - Dump op CFG pass -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/ControlFlow.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir {
#define GEN_PASS_DEF_DUMPOPCFG
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct DumpOpCFGPass : public impl::DumpOpCFGBase<DumpOpCFGPass> {
  void runOnOperation() override;
};
} // namespace

static CFGFlowPoint *buildFlowPoint(Block *block, CFGContext &context) {}

// static CFGOp *buildOpCFG(FunctionOpInterface op, CFGContext &context) {
//   if (op.getFunctionBody().empty())
//     return nullptr;
//   auto cfgOp = new CFGOp(op);
//   context.insert(cfgOp);
//   cfgOp.addSuccessor(buildFlowPoint(&cfgOp.front()));
//   return cfgOp;
// }

void DumpOpCFGPass::runOnOperation() {
  CFGContext cfgContext;
  // if (CFGOp *cfg = ::buildOpCFG(getOperation(), cfgContext))
  // cfg->dump();
  llvm::errs() << sizeof(llvm::iplist<Operation>) << "\n";
}
