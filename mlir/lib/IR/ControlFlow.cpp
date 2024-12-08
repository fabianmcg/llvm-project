//===- ControlFlow.h - MLIR CFG utilities ------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines CFG classes.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/ControlFlow.h"
#include "llvm/Support/ScopedPrinter.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// CFGOperand
//===----------------------------------------------------------------------===//

CFGOperand::CFGOperand(CFGTerminator *owner, CFGFlowPoint *point)
    : Base(owner, point) {}

IRObjectWithUseList<CFGOperand, CFGTerminator> *
CFGOperand::getUseList(CFGFlowPoint *point) {
  return point;
}

unsigned CFGOperand::getOperandNumber() {
  return this - &getOwner()->getSuccessorOperands()[0];
}

//===----------------------------------------------------------------------===//
// CFGPredecessorIterator
//===----------------------------------------------------------------------===//

CFGTerminator *CFGPredecessorIterator::unwrap(CFGOperand &value) {
  return value.getOwner();
}

unsigned CFGPredecessorIterator::getSuccessorIndex() const {
  return I->getOperandNumber();
}

//===----------------------------------------------------------------------===//
// CFGSuccessorRange
//===----------------------------------------------------------------------===//

CFGSuccessorRange::CFGSuccessorRange() : CFGSuccessorRange(nullptr, 0) {}
CFGSuccessorRange::CFGSuccessorRange(CFGTerminator *point)
    : CFGSuccessorRange() {
  if ((count = point->getNumSuccessors()))
    base = point->getSuccessorOperands().data();
}

//===----------------------------------------------------------------------===//
// CFGLabelAttr
//===----------------------------------------------------------------------===//

WalkResult CFGLabelAttr::walk(function_ref<WalkResult(CFGLabel)> walkFn) {
  AttrTypeWalker walker;
  // Walk label, but skip any other attribute.
  walker.addWalk([&](Attribute attr) {
    if (auto loc = llvm::dyn_cast<CFGLabelAttr>(attr))
      return walkFn(loc);

    return WalkResult::skip();
  });
  return walker.walk<WalkOrder::PreOrder>(*this);
}

/// Methods for support type inquiry through isa, cast, and dyn_cast.
bool CFGLabelAttr::classof(Attribute attr) {
  return attr.hasTrait<AttributeTrait::IsCFGLabel>();
}
