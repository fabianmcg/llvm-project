//===-- SPIRVStripConvergentIntrinsics.cpp ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass trims convergence intrinsics as those were only useful when
// modifying the CFG during IR passes.
//
//===----------------------------------------------------------------------===//

#include "SPIRV.h"
#include "SPIRVSubtarget.h"
#include "SPIRVUtils.h"
#include "llvm/CodeGen/IntrinsicLowering.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LowerMemIntrinsics.h"

using namespace llvm;

namespace {
class SPIRVStripConvergentIntrinsics : public FunctionPass {
public:
  static char ID;

  SPIRVStripConvergentIntrinsics() : FunctionPass(ID) {}

  virtual bool runOnFunction(Function &F) override {
    DenseSet<Instruction *> ToRemove;

    // Is the instruction is a convergent intrinsic, add it to kill-list and
    // returns true. Returns false otherwise.
    auto CleanupIntrinsic = [&](IntrinsicInst *II) {
      if (II->getIntrinsicID() != Intrinsic::experimental_convergence_entry &&
          II->getIntrinsicID() != Intrinsic::experimental_convergence_loop &&
          II->getIntrinsicID() != Intrinsic::experimental_convergence_anchor)
        return false;

      II->replaceAllUsesWith(UndefValue::get(II->getType()));
      ToRemove.insert(II);
      return true;
    };

    // Replace the given CallInst by a similar CallInst with no convergencectrl
    // attribute.
    auto CleanupCall = [&](CallInst *CI) {
      auto OB = CI->getOperandBundle(LLVMContext::OB_convergencectrl);
      if (!OB.has_value())
        return;

      auto *NewCall = CallBase::removeOperandBundle(
          CI, LLVMContext::OB_convergencectrl, CI->getIterator());
      NewCall->copyMetadata(*CI);
      CI->replaceAllUsesWith(NewCall);
      ToRemove.insert(CI);
    };

    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (auto *II = dyn_cast<IntrinsicInst>(&I))
          if (CleanupIntrinsic(II))
            continue;
        if (auto *CI = dyn_cast<CallInst>(&I))
          CleanupCall(CI);
      }
    }

    // All usages must be removed before their definition is removed.
    for (Instruction *I : ToRemove)
      I->eraseFromParent();

    return ToRemove.size() != 0;
  }
};
} // namespace

char SPIRVStripConvergentIntrinsics::ID = 0;
INITIALIZE_PASS(SPIRVStripConvergentIntrinsics, "strip-convergent-intrinsics",
                "SPIRV strip convergent intrinsics", false, false)

FunctionPass *llvm::createSPIRVStripConvergenceIntrinsicsPass() {
  return new SPIRVStripConvergentIntrinsics();
}
