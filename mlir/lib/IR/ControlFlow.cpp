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

static CFGPointWithSuccessors *getAsPointWithSuccessors(CFGPoint *point) {
  if (auto term = dyn_cast<CFGTerminator>(point))
    return term;
  if (auto op = dyn_cast<CFGOp>(point))
    return op;
  llvm_unreachable("invalid operand owner");
}

unsigned CFGOperand::getOperandNumber() {
  return this -
         &getAsPointWithSuccessors(getOwner())->getSuccessorOperands()[0];
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
// Control-flow region
//===----------------------------------------------------------------------===//

CFGRegion::~CFGRegion() { dropAllUses(); }

//===----------------------------------------------------------------------===//
// Control-flow op
//===----------------------------------------------------------------------===//

CFGOp::CFGOp(Operation *op) : CFGFlowPoint(op) {
  if (op->getNumRegions() == 0)
    return;
  regions = std::allocator<CFGRegion>().allocate(op->getNumRegions());
  for (Region &region : op->getRegions())
    new (regions + region.getRegionNumber()) CFGRegion(this, &region);
}

CFGOp::~CFGOp() {
  dropAllUses();
  for (CFGRegion &region : getRegions())
    region.dropAllUses();
  std::allocator<CFGRegion>().deallocate(regions, getOp()->getNumRegions());
}

//===----------------------------------------------------------------------===//
// Control-flow graph context
//===----------------------------------------------------------------------===//

void CFGContext::PointDeleter::operator()(CFGPoint *ptr) const {
  if (auto point = dyn_cast<CFGTerminator>(ptr))
    delete point;
  else if (auto point = dyn_cast<CFGOp>(ptr)) {
    delete point;
  }
}

CFGRegion *CFGContext::lookup(Region *region) const {
  auto op = dyn_cast_or_null<CFGOp>(lookup(region->getParentOp()));
  if (!op)
    return nullptr;
  return &op->getRegions()[region->getRegionNumber()];
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
