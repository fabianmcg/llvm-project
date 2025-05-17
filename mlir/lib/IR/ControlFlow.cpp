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
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ScopedPrinter.h"

using namespace mlir;

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

//===----------------------------------------------------------------------===//
// CFGOperand
//===----------------------------------------------------------------------===//

CFGOperand::CFGOperand(CFGOperation *owner, CFGFlowPoint *point)
    : Base(owner, point) {}

IRObjectWithUseList<CFGOperand, CFGOperation> *
CFGOperand::getUseList(CFGFlowPoint *point) {
  return point;
}

unsigned CFGOperand::getOperandNumber() {
  return this - &(getOwner()->getSuccessorOperands()[0]);
}

//===----------------------------------------------------------------------===//
// CFGPredecessorIterator
//===----------------------------------------------------------------------===//

CFGOperation *CFGPredecessorIterator::unwrap(CFGOperand &value) {
  return value.getOwner();
}

unsigned CFGPredecessorIterator::getSuccessorIndex() const {
  return I->getOperandNumber();
}

//===----------------------------------------------------------------------===//
// CFGSuccessorRange
//===----------------------------------------------------------------------===//

CFGSuccessorRange::CFGSuccessorRange() : CFGSuccessorRange(nullptr, 0) {}
CFGSuccessorRange::CFGSuccessorRange(CFGOperation *op) : CFGSuccessorRange() {
  if ((count = op->getNumSuccessors()))
    base = op->getSuccessorOperands().data();
}

//===----------------------------------------------------------------------===//
// CFGFlowPoint
//===----------------------------------------------------------------------===//

CFGFlowPoint::~CFGFlowPoint() { dropAllUses(); }

//===----------------------------------------------------------------------===//
// CFGBlock
//===----------------------------------------------------------------------===//

bool CFGBlock::classof(CFGPoint const *point) {
  return isa<CFGRegion *>(point->getParent());
}

CFGRegion *CFGBlock::getParent() const {
  return cast_if_present<CFGRegion *>(parent);
}

//===----------------------------------------------------------------------===//
// CFGRegion
//===----------------------------------------------------------------------===//

bool CFGRegion::classof(CFGPoint const *point) {
  return isa<CFGOperation *>(point->getParent());
}

CFGOperation *CFGRegion::getParent() const {
  return cast_if_present<CFGOperation *>(parent);
}

//===----------------------------------------------------------------------===//
// CFGOperation
//===----------------------------------------------------------------------===//

CFGOperation::CFGOperation(Operation *op, bool isTerminator)
    : opIsTerm(op, isTerminator) {
  regions = std::allocator<CFGRegion>().allocate(getOp()->getNumRegions());
  for (unsigned i = 0, numRegions = getOp()->getNumRegions(); i < numRegions;
       ++i) {
    new (regions + i) CFGRegion(this, getOp()->getRegion(i));
  }
}

CFGOperation::~CFGOperation() {
  if (regions) {
    std::destroy_n(regions, getOp()->getNumRegions());
    std::allocator<CFGRegion>().deallocate(regions, getOp()->getNumRegions());
    regions = nullptr;
  }
  opIsTerm.setPointerAndInt(nullptr, false);
  successors.clear();
}

CFGBlock *CFGOperation::getParent() const {
  if (auto op = dyn_cast<const CFGOp>(this))
    return cast_if_present<CFGBlock *>(op->parent);
  return cast_if_present<CFGBlock *>(cast<const CFGTerminator>(this)->parent);
}

void CFGOperation::setParent(CFGBlock *block) {
  if (auto op = dyn_cast<CFGOp>(this))
    op->parent = block;
  else
    cast<CFGTerminator>(this)->parent = block;
}

void CFGOperation::appendSuccessors(ArrayRef<CFGFlowPoint *> successors) {
  this->successors.reserve(this->successors.size() + successors.size());
  for (CFGFlowPoint *point : successors) {
    assert(point && "null point detected");
    this->successors.push_back(CFGOperand(this, point));
  }
  // TODO: Make this efficient, here for debugging.
  std::sort(this->successors.begin(), this->successors.end(),
            [](const CFGOperand &lhs, const CFGOperand &rhs) {
              return lhs.get() < rhs.get();
            });
  auto it = std::unique(this->successors.begin(), this->successors.end(),
                        [](const CFGOperand &lhs, const CFGOperand &rhs) {
                          return lhs.get() == rhs.get();
                        });
  this->successors.erase(it, this->successors.end());
}

//===----------------------------------------------------------------------===//
// llvm::ilist_traits
//===----------------------------------------------------------------------===//

void llvm::ilist_traits<::mlir::CFGOperation>::deleteNode(CFGOperation *op) {
  if (auto cfOp = dyn_cast<CFGOp>(op)) {
    delete cfOp;
    return;
  }
  auto *termOp = cast<CFGTerminator>(op);
  delete termOp;
}

CFGBlock *llvm::ilist_traits<::mlir::CFGOperation>::getContainingBlock() {
  size_t offset(
      size_t(&((CFGBlock *)nullptr->*CFGBlock::getSublistAccess(nullptr))));
  iplist<CFGOperation> *anchor(static_cast<iplist<CFGOperation> *>(this));
  return reinterpret_cast<CFGBlock *>(reinterpret_cast<char *>(anchor) -
                                      offset);
}

/// This is a trait method invoked when an operation is added to a block.  We
/// keep the block pointer up to date.
void llvm::ilist_traits<::mlir::CFGOperation>::addNodeToList(CFGOperation *op) {
  assert(!op->getParent() && "already in an operation block!");
  op->setParent(getContainingBlock());
}

/// This is a trait method invoked when an operation is removed from a block.
/// We keep the block pointer up to date.
void llvm::ilist_traits<::mlir::CFGOperation>::removeNodeFromList(
    CFGOperation *op) {
  assert(op->getParent() && "not already in an operation block!");
  op->setParent(nullptr);
}

/// This is a trait method invoked when an operation is moved from one block
/// to another.  We keep the block pointer up to date.
void llvm::ilist_traits<::mlir::CFGOperation>::transferNodesFromList(
    ilist_traits<CFGOperation> &otherList, op_iterator first,
    op_iterator last) {
  CFGBlock *curParent = getContainingBlock();

  // If we are transferring operations within the same block, the block
  // pointer doesn't need to be updated.
  if (curParent == otherList.getContainingBlock())
    return;

  // Update the 'block' member of each operation.
  for (; first != last; ++first)
    first->setParent(curParent);
}

CFGRegion *llvm::ilist_traits<::mlir::CFGBlock>::getParentRegion() {
  size_t offset(
      size_t(&((CFGRegion *)nullptr->*CFGRegion::getSublistAccess(nullptr))));
  iplist<CFGBlock> *anchor(static_cast<iplist<CFGBlock> *>(this));
  return reinterpret_cast<CFGRegion *>(reinterpret_cast<char *>(anchor) -
                                       offset);
}

/// This is a trait method invoked when a basic block is added to a region.
/// We keep the region pointer up to date.
void llvm::ilist_traits<::mlir::CFGBlock>::addNodeToList(CFGBlock *block) {
  assert(!block->getParent() && "already in a region!");
  block->parent = getParentRegion();
}

/// This is a trait method invoked when an operation is removed from a
/// region.  We keep the region pointer up to date.
void llvm::ilist_traits<::mlir::CFGBlock>::removeNodeFromList(CFGBlock *block) {
  assert(block->getParent() && "not already in a region!");
  block->parent = nullptr;
}

/// This is a trait method invoked when an operation is moved from one block
/// to another.  We keep the block pointer up to date.
void llvm::ilist_traits<::mlir::CFGBlock>::transferNodesFromList(
    ilist_traits<CFGBlock> &otherList, block_iterator first,
    block_iterator last) {
  // If we are transferring operations within the same function, the parent
  // pointer doesn't need to be updated.
  auto *curParent = getParentRegion();
  if (curParent == otherList.getParentRegion())
    return;

  // Update the 'parent' member of each CFGBlock.
  for (; first != last; ++first)
    first->parent = curParent;
}

//===----------------------------------------------------------------------===//
// API
//===----------------------------------------------------------------------===//

/// Handle and create terminators.
static CFGTerminator *handleTerminator(Operation *op, CFGContext &context) {
  if (auto *term = cast_if_present<CFGTerminator>(context.lookupTerminator(op)))
    return term;
  CFGTerminator *term = new CFGTerminator(op);
  context.insert(term);
  if (auto cfgTerm = dyn_cast<CFGTerminatorOpInterface>(op)) {
    SmallVector<CFGLabel> labels;
    cfgTerm.getSuccessorLabels(std::nullopt, labels);
    context.appendUnresolvedLabels(cfgTerm, labels);
  } else if (op->hasSuccessors()) {
    for (Block *succ : op->getSuccessors()) {
      CFGBlock *cfgBlock = context.lookup(succ);
      assert(cfgBlock && "ill-formed CFG");
      term->appendSuccessors({static_cast<CFGFlowPoint *>(cfgBlock)});
    }
  } else {
    llvm_unreachable("invalid terminator");
  }
  return term;
}

CFGOp *mlir::buildOpCFG(CFGOpInterface op, CFGContext &context) {
  if (auto *cfOp = cast_if_present<CFGOp>(context.lookupOp(op)))
    return cfOp;
  CFGOp *cfOp = new CFGOp(op);
  context.insert(cfOp);
  // Check for unresolved terminators accepted by this operation.
  SmallVector<std::pair<CFGLabel, CFGTerminatorOpInterface>>
      acceptedTerminators;
  // Build the nested ops CFG.
  for (CFGRegion &region : cfOp->getRegionsMutable()) {
    context.insert(&region);
    // Add all blocks to the region so that we can resolve block successors.
    for (Block &block : *region.getRegion()) {
      // Create a new block and add it to the region.
      auto *cfBlock = new CFGBlock(block);
      region.push_back(cfBlock);
      context.insert(cfBlock);
    }
    for (CFGBlock &block : region.getBlocks()) {
      // Create the CFG for the nested ops in the block.
      for (auto op : block.getBlock()->getOps<CFGOpInterface>())
        block.push_back(buildOpCFG(op, context));
      // Handle the terminator.
      block.push_back(
          handleTerminator(block.getBlock()->getTerminator(), context));
    }
    op.getAcceptedTerminators(context.getUnresolvedTerminators(),
                              acceptedTerminators);
    context.eraseUnresolvedTerminators(acceptedTerminators);
  }
  // Get the op on entry successors.
  SmallVector<CFGSuccessor> onEntrySuccessors;
  op.getOnEntrySuccessors(std::nullopt, onEntrySuccessors);
  SmallVector<CFGFlowPoint *> successors;
  // Add the on entry successors.
  for (const CFGSuccessor &succ : onEntrySuccessors) {
    if (succ.getSuccessor().isParent()) {
      successors.push_back(cfOp);
    } else if (Region *region = succ.getSuccessor().getRegionOrNull()) {
      successors.push_back(cfOp->getRegion(region->getRegionNumber()));
    }
  }
  cfOp->appendSuccessors(successors);
  // Add the accepted terminators successors.
  for (auto [label, term] : acceptedTerminators) {
    onEntrySuccessors.clear();
    op.getLabelSuccessors(label, onEntrySuccessors);
    for (CFGSuccessor &succ : onEntrySuccessors) {
      auto *cfTerm = cast<CFGTerminator>(context.lookupTerminator(term));
      if (succ.getSuccessor().isParent()) {
        cfTerm->appendSuccessors({static_cast<CFGFlowPoint *>(cfOp)});
      } else if (Region *region = succ.getSuccessor().getRegionOrNull()) {
        cfTerm->appendSuccessors({static_cast<CFGFlowPoint *>(
            cfOp->getRegion(region->getRegionNumber()))});
      }
    }
  }
  return cfOp;
}

//===----------------------------------------------------------------------===//
// BuiltinDialect
//===----------------------------------------------------------------------===//

void BuiltinDialect::registerControlFlowAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/IR/BuiltinControlFlowAttributes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// CFG IncGen
//===----------------------------------------------------------------------===//
#define GET_ATTRDEF_CLASSES
#include "mlir/IR/BuiltinControlFlowAttributes.cpp.inc"

#include "mlir/IR/ControlFlowGraphInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// CFGPrinter
//===----------------------------------------------------------------------===//

namespace {
struct CFGPrinter {
  CFGPrinter(llvm::raw_ostream &stream) : stream(stream) {}

  void dumpGraph(CFGOp *op);

private:
  /// Dump a generic CFG op.
  void dump(CFGOperation *op);
  /// Dump a CFG op.
  void dump(CFGOp *op);
  /// Dump a CFG terminator op.
  void dump(CFGTerminator *op);
  /// Dump a region.
  void dump(CFGRegion *region, uint32_t opId, unsigned regionNumber);
  /// Dump a block.
  void dump(CFGBlock *block, CFGOperation *parentOp, uint32_t opId,
            unsigned regionNumber, uint32_t blockId);
  /// Print an indented line.
  llvm::raw_ostream &print() {
    stream.printIndent();
    return stream.getOStream();
  }

  /// Return a unique ID for ptr.
  uint32_t getId(void *ptr) {
    uint32_t &label = labels[ptr];
    if (label == 0)
      label = labels.size();
    return label;
  }

  llvm::ScopedPrinter stream;
  llvm::DenseMap<void *, uint32_t> labels;
};
} // namespace

void CFGPrinter::dumpGraph(CFGOp *op) {
  // Dump the header
  print() << "digraph {\n";
  stream.indent();
  print() << "rankdir=LR;\n";
  print() << "node [ shape=record ];\n";
  // Start dumping the graph
  dump(op);
  // Dump the epilogue
  stream.unindent();
  print() << "}\n";
}

void CFGPrinter::dump(CFGOperation *op) {
  if (auto cfOp = dyn_cast<CFGOp>(op))
    return dump(cfOp);
  dump(cast<CFGTerminator>(op));
}

void CFGPrinter::dump(CFGOp *op) {
  // Get an unique ID and label for the op.
  uint32_t opId = getId(op->getOp());
  // Print the node.
  print() << opId << " [\n";
  stream.indent();
  // Print the label: Op[`op name`, ptr]
  print() << "label = \"<root>Op[`" << op->getOp()->getName().getStringRef()
          << "`, " << op->getOp() << "]\"\n";
  stream.indent();
  // Print the op's regions.
  for (CFGRegion &region : op->getRegionsMutable()) {
    // Print an entry for each region.
    unsigned regionNumber = region.getRegion()->getRegionNumber();
    print() << "+ \"| <R" << regionNumber << ">Region: " << regionNumber
            << "\"\n";
  }
  // Print the op's successors.
  print() << "+ \"| Successors: ";
  llvm::interleaveComma(
      CFGSuccessorRange(op), stream.getOStream(), [&](CFGFlowPoint *point) {
        if (auto cfOp = dyn_cast<CFGOp>(point)) {
          assert(cfOp == op && "ill-formed CFG");
          stream.getOStream() << "self";
          return;
        }
        if (auto cfRegion = dyn_cast<CFGRegion>(point)) {
          assert(cfRegion->getParent() == op && "ill-formed CFG");
          stream.getOStream()
              << "Region[" << cfRegion->getRegion()->getRegionNumber() << "]";
          return;
        }
        llvm_unreachable("ill-formed CFG");
      });
  stream.getOStream() << "\"\n";
  // Finish the node.
  stream.unindent(2);
  print() << "];\n";
  // Dump the regions.
  for (CFGRegion &region : op->getRegionsMutable()) {
    if (region.empty() || region.front().empty())
      continue;
    unsigned regionNumber = region.getRegion()->getRegionNumber();
    dump(&region, opId, regionNumber);
  }
}

void CFGPrinter::dump(CFGTerminator *op) {
  // Get an unique ID and label for the op.
  uint32_t opId = getId(op->getOp());
  // Print the node.
  print() << opId << " [\n";
  stream.indent();
  // Print the label: Op[`op name`, ptr]
  print() << "label = \"<root>Op[`" << op->getOp()->getName().getStringRef()
          << "`, " << op->getOp() << "]\"\n";
  stream.indent();
  // Print the op's regions.
  for (CFGRegion &region : op->getRegionsMutable()) {
    // Print an entry for each region.
    unsigned regionNumber = region.getRegion()->getRegionNumber();
    print() << "+ \"| <R" << regionNumber << ">Region: " << regionNumber
            << "\"\n";
  }
  print() << "+ \"| <succ>Terminator\"\n";
  // Finish the node.
  stream.unindent(2);
  print() << "];\n";
  // Dump the regions.
  for (CFGRegion &region : op->getRegionsMutable()) {
    if (region.empty() || region.front().empty())
      continue;
    unsigned regionNumber = region.getRegion()->getRegionNumber();
    dump(&region, opId, regionNumber);
  }
  // Print the op's successors.
  for (CFGFlowPoint *point : CFGSuccessorRange(op)) {
    if (auto cfOp = dyn_cast<CFGOp>(point)) {
      uint32_t tgtOpId = getId(cfOp->getOp());
      print() << opId << ":" << "succ -> " << tgtOpId << ":root;\n";
      continue;
    }
    if (auto cfRegion = dyn_cast<CFGRegion>(point)) {
      uint32_t tgtOpId = getId(cfRegion->getParent()->getOp());
      print() << opId << ":" << "succ -> " << tgtOpId << ":R"
              << cfRegion->getRegion()->getRegionNumber() << ";\n";
      continue;
    }
    if (auto cfBlock = dyn_cast<CFGBlock>(point)) {
      uint32_t tgtId = getId(cfBlock);
      print() << opId << ":" << "succ -> " << tgtId << ":root;\n";
      continue;
    }
    llvm_unreachable("ill-formed CFG");
  }
}

void CFGPrinter::dump(CFGRegion *region, uint32_t opId, unsigned regionNumber) {
  if (region->empty())
    return;
  uint32_t entryBlockId = getId(region->front().getBlock());
  print() << opId << ":" << "R" << regionNumber << " -> " << entryBlockId
          << ":root;\n";
  // Print the blocks.
  uint32_t counter = 0;
  for (CFGBlock &block : region->getBlocks())
    dump(&block, region->getParent(), opId, regionNumber, counter++);
}

void CFGPrinter::dump(CFGBlock *block, CFGOperation *parentOp, uint32_t opId,
                      unsigned regionNumber, uint32_t blockId) {
  // Get an unique ID for the region.
  uint32_t bid = getId(block->getBlock());
  // Print the node.
  print() << bid << " [\n";
  stream.indent();
  // Print the label: Block[op = address, region = number, block = number]
  print() << "label = \"<root>Block[op = " << parentOp->getOp()
          << ", region = " << regionNumber << ", block = " << blockId
          << "]\"\n";
  stream.indent();
  // Print the block's ops.
  uint32_t counter = 0;
  for (CFGOperation &op : block->getOperations()) {
    // Print an entry for each block.
    print() << "+ \"| <O" << counter++ << ">Op[`"
            << op.getOp()->getName().getStringRef() << "`, " << op.getOp()
            << "]\"\n";
  }
  stream.unindent(2);
  print() << "];\n";
  // Print the ops.
  counter = 0;
  for (CFGOperation &op : block->getOperations()) {
    uint32_t opId = getId(op.getOp());
    dump(&op);
    // Print the edge connecting the block to the op.
    print() << bid << ":" << "O" << counter++ << " -> " << opId << ":root;\n";
  }
}

void CFGOp::print(llvm::raw_ostream &stream) const {
  CFGPrinter(stream).dumpGraph(const_cast<CFGOp *>(this));
}
