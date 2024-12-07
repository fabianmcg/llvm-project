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

CFGOperand::CFGOperand(CFGPoint *owner, CFGFlowPoint *point)
    : Base(owner, point) {}

IRObjectWithUseList<CFGOperand, CFGPoint> *
CFGOperand::getUseList(CFGFlowPoint *point) {
  return point;
}

CFGPointWithSuccessors *CFGOperand::getOwnerAsPointWithSuccessors() const {
  CFGPoint *owner = getOwner();
  if (auto point = dyn_cast<CFGTerminator>(owner))
    return point;
  else if (auto point = dyn_cast<CFGOp>(owner))
    return point;
  llvm_unreachable("invalid CFG point");
}

unsigned CFGOperand::getOperandNumber() {
  return this - &getOwnerAsPointWithSuccessors()->getSuccessorOperands()[0];
}

//===----------------------------------------------------------------------===//
// CFGPredecessorIterator
//===----------------------------------------------------------------------===//

CFGPoint *CFGPredecessorIterator::unwrap(CFGOperand &value) {
  return value.getOwner();
}

unsigned CFGPredecessorIterator::getSuccessorIndex() const {
  return I->getOperandNumber();
}

//===----------------------------------------------------------------------===//
// CFGSuccessorRange
//===----------------------------------------------------------------------===//

CFGSuccessorRange::CFGSuccessorRange() : CFGSuccessorRange(nullptr, 0) {}
CFGSuccessorRange::CFGSuccessorRange(CFGPointWithSuccessors *point)
    : CFGSuccessorRange() {
  if ((count = point->getNumSuccessors()))
    base = point->getSuccessorOperands().data();
}

//===----------------------------------------------------------------------===//
// Control-flow operation
//===----------------------------------------------------------------------===//

namespace {
struct CFGDumper {
  CFGDumper(llvm::raw_ostream &os) : os(os), logger(os) {}

  void dump(CFGOp *point);
  void dump(CFGFlowPoint *point);
  void dump(CFGTerminator *point);

  void dumpHeader(CFGPoint *point);

  llvm::raw_ostream &os;
  llvm::ScopedPrinter printer;
  DenseMap<CFGPoint *, int> point2Labels;
};
} // namespace

void CFGDumper::dump(CFGOp *point) {
  dumpHeader(point);
  printer.indent();
  for (CFGFlowPoint *successor : CFGSuccessorRange(point))
    dump(successor);
  printer.unindent();
}

void CFGDumper::dump(CFGFlowPoint *point) { dumpHeader(point); }

void CFGOp::print(llvm::raw_ostream &os) const {
  CFGDumper(llvm::errs()).dump(const_cast<CFGOp*>(this));
}

void CFGOp::dump() const { print(llvm::errs()); }

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
