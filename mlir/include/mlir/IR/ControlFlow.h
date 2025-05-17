//===- ControlFlow.h - MLIR CFG utilities ------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares CFG classes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_CONTROLFLOW_H
#define MLIR_IR_CONTROLFLOW_H

#include "mlir/IR/ControlFlowLabel.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"

namespace mlir {
//===----------------------------------------------------------------------===//
// CFGBranchPoint
//===----------------------------------------------------------------------===//
/// This class represents a successor point to a `CFGOpInterface` operation.
class CFGBranchPoint : public llvm::PointerUnion<Region *, Attribute> {
private:
  using Base = llvm::PointerUnion<Region *, Attribute>;
  using Base::Base;
  // Private constructor to encourage the use of `CFGBranchPoint::parent`.
  CFGBranchPoint() : Base(nullptr) {}

public:
  using Base::operator=;
  CFGBranchPoint(Region &region) : Base(&region) {}
  CFGBranchPoint(SymbolRefAttr symbol) : Base(symbol) {
    assert(symbol && "the symbol attribute cannot be null");
  }
  CFGBranchPoint(FlatSymbolRefAttr symbol) : Base(symbol) {
    assert(symbol && "the symbol attribute cannot be null");
  }
  CFGBranchPoint(std::nullptr_t) = delete;
  /// Returns an instance of `CFGBranchPoint` representing the parent
  /// operation.
  static CFGBranchPoint parent() {
    CFGBranchPoint point;
    return point;
  }
  /// Returns the held region or null if it's not a region.
  Region *getRegionOrNull() const {
    return dyn_cast_if_present<Region *>(static_cast<const Base &>(*this));
  }
  /// Returns the held attribute or null if it's not an attribute.
  Attribute getAttrOrNull() const {
    return dyn_cast_if_present<Attribute>(static_cast<const Base &>(*this));
  }
  /// Return true if the successor is the parent operation.
  bool isParent() const { return isNull(); }

  /// Compares the branch point to `other`.
  bool operator==(CFGBranchPoint other) const {
    return getOpaqueValue() == other.getOpaqueValue();
  }
};

//===----------------------------------------------------------------------===//
// CFGSuccessor
//===----------------------------------------------------------------------===//
/// This class represents a successor to a `CFGOpInterface` operation.
class CFGSuccessor {
public:
  /// Initialize a successor that branches to another point.
  CFGSuccessor(CFGBranchPoint point, ValueRange inputs = {})
      : point(point), inputs(inputs) {}

  /// Return the given successor.
  CFGBranchPoint getSuccessor() const { return point; }

  /// Return the inputs to the successor that are remapped by the exit values of
  /// the current region.
  ValueRange getSuccessorInputs() const { return inputs; }

private:
  CFGBranchPoint point;
  ValueRange inputs;
};

class CFGFlowPoint;
class CFGBlock;
class CFGRegion;
class CFGOperation;
class CFGOp;
class CFGTerminator;
class CFGTerminatorOpInterface;
} // namespace mlir

#define GET_ATTRDEF_CLASSES
#include "mlir/IR/BuiltinControlFlowAttributes.h.inc"

#include "mlir/IR/ControlFlowGraphInterfaces.h.inc"

namespace llvm {
//===----------------------------------------------------------------------===//
// ilist_traits for CFGOperation
//===----------------------------------------------------------------------===//
template <>
struct ilist_traits<::mlir::CFGOperation> {
  using CFGOperation = ::mlir::CFGOperation;
  using op_iterator = simple_ilist<CFGOperation>::iterator;

  static void deleteNode(CFGOperation *op);
  void addNodeToList(CFGOperation *op);
  void removeNodeFromList(CFGOperation *op);
  void transferNodesFromList(ilist_traits<CFGOperation> &otherList,
                             op_iterator first, op_iterator last);

private:
  mlir::CFGBlock *getContainingBlock();
};

//===----------------------------------------------------------------------===//
// ilist_traits for CFGBlock
//===----------------------------------------------------------------------===//
template <>
struct ilist_traits<::mlir::CFGBlock>
    : public ilist_alloc_traits<::mlir::CFGBlock> {
  using CFGBlock = ::mlir::CFGBlock;
  using block_iterator = simple_ilist<::mlir::CFGBlock>::iterator;

  void addNodeToList(CFGBlock *block);
  void removeNodeFromList(CFGBlock *block);
  void transferNodesFromList(ilist_traits<CFGBlock> &otherList,
                             block_iterator first, block_iterator last);

private:
  mlir::CFGRegion *getParentRegion();
};
} // namespace llvm

namespace mlir {
//===----------------------------------------------------------------------===//
// CFGOperand
//===----------------------------------------------------------------------===//
/// A control-flow operand. This class allows CFG operations to indicate its
/// successors.
class CFGOperand : public IROperand<CFGOperand, CFGFlowPoint *, CFGOperation> {
public:
  using Base = IROperand<CFGOperand, CFGFlowPoint *, CFGOperation>;

  /// Provide the use list that is attached to the given operation.
  static IRObjectWithUseList<CFGOperand, CFGOperation> *
  getUseList(CFGFlowPoint *point);

  /// Return which operand this is in the operand list of the point.
  unsigned getOperandNumber();

private:
  friend class CFGOperation;
  CFGOperand(CFGOperation *owner, CFGFlowPoint *point);
};

//===----------------------------------------------------------------------===//
// Predecessors
//===----------------------------------------------------------------------===//
/// Implement a predecessor iterator for control-flow points.
class CFGPredecessorIterator final
    : public llvm::mapped_iterator<ValueUseIterator<CFGOperand, CFGOperation>,
                                   CFGOperation *(*)(CFGOperand &)> {
  static CFGOperation *unwrap(CFGOperand &value);

public:
  /// Initializes the operand type iterator to the specified operand  iterator
  CFGPredecessorIterator(ValueUseIterator<CFGOperand, CFGOperation> it)
      : llvm::mapped_iterator<ValueUseIterator<CFGOperand, CFGOperation>,
                              CFGOperation *(*)(CFGOperand &)>(it, &unwrap) {}

  explicit CFGPredecessorIterator(CFGOperand *operand)
      : CFGPredecessorIterator(
            ValueUseIterator<CFGOperand, CFGOperation>(operand)) {}

  /// Get the successor number in the predecessor.
  unsigned getSuccessorIndex() const;
};

//===----------------------------------------------------------------------===//
// Successors
//===----------------------------------------------------------------------===//
/// This class implements the successor iterators for control-flow points.
class CFGSuccessorRange final
    : public llvm::detail::indexed_accessor_range_base<
          CFGSuccessorRange, CFGOperand *, CFGFlowPoint *, CFGFlowPoint *,
          CFGFlowPoint *> {
public:
  using RangeBaseT::RangeBaseT;
  CFGSuccessorRange();
  CFGSuccessorRange(CFGOperation *op);

private:
  /// See `llvm::detail::indexed_accessor_range_base` for details.
  static CFGOperand *offset_base(CFGOperand *object, ptrdiff_t index) {
    return object + index;
  }

  /// See `llvm::detail::indexed_accessor_range_base` for details.
  static CFGFlowPoint *dereference_iterator(CFGOperand *object,
                                            ptrdiff_t index) {
    return object[index].get();
  }

  friend RangeBaseT;
};

//===----------------------------------------------------------------------===//
// CFGPoint
//===----------------------------------------------------------------------===//
/// Base class for all control-flow points. This class allows identifying the
/// CFG point kind via its parent. It's important to note that this class is not
/// meant to be used directly, use its subclasses instead.
class CFGPoint {
public:
  using CFGPointTy =
      llvm::PointerUnion<CFGBlock *, CFGRegion *, CFGOperation *>;

  CFGPointTy getParent() const { return parent; }

protected:
  CFGPointTy parent;
};

//===----------------------------------------------------------------------===//
// CFGPoint
//===----------------------------------------------------------------------===//
/// A control-flow point accepting control from a CFG operation.
class CFGFlowPoint : public CFGPoint,
                     public IRObjectWithUseList<CFGOperand, CFGOperation> {
public:
  ~CFGFlowPoint();
  //===--------------------------------------------------------------------===//
  // Predecessors.
  //===--------------------------------------------------------------------===//
  /// Return true if this point has predecessors.
  bool hasPredecessors() { return pred_begin() != pred_end(); }

  // Predecessor iteration.
  using pred_iterator = CFGPredecessorIterator;

  pred_iterator pred_begin() {
    return pred_iterator((CFGOperand *)getFirstUse());
  }

  pred_iterator pred_end() { return pred_iterator(nullptr); }

  iterator_range<pred_iterator> getPredecessors() {
    return {pred_begin(), pred_end()};
  }
};

//===----------------------------------------------------------------------===//
// CFGBlock
//===----------------------------------------------------------------------===//
/// A control-flow block. This class accepts control-flow from the parent region
/// or a CFG operation. Further this class contains a list of all the CFG
/// operations directly under it.
class CFGBlock : public CFGFlowPoint,
                 public llvm::ilist_node_with_parent<CFGBlock, CFGRegion> {
public:
  CFGBlock(Block &block) : block(&block) {}
  using OpListTy = llvm::iplist<CFGOperation>;
  static bool classof(CFGPoint const *point);

  /// Returns the parent CFG region.
  CFGRegion *getParent() const;

  /// Returns the associated block with this CFG point.
  Block *getBlock() const { return block; }

  /// Returns the op list.
  OpListTy &getOperations() { return operations; }

  // Iteration over the operations in the block.
  using iterator = OpListTy::iterator;
  using reverse_iterator = OpListTy::reverse_iterator;

  iterator begin() { return operations.begin(); }
  iterator end() { return operations.end(); }

  reverse_iterator rbegin() { return operations.rbegin(); }
  reverse_iterator rend() { return operations.rend(); }

  bool empty() { return operations.empty(); }
  void push_back(CFGOperation *op) { operations.push_back(op); }
  void push_front(CFGOperation *op) { operations.push_front(op); }

  CFGOperation &back() { return operations.back(); }
  CFGOperation &front() { return operations.front(); }

  /// Returns the terminator operation.
  CFGTerminator &getTerminator();

  /// Returns pointer to member of operation list.
  static OpListTy CFGBlock::*getSublistAccess(CFGOperation *) {
    return &CFGBlock::operations;
  }

private:
  Block *block;
  OpListTy operations;

  CFGBlock(CFGBlock &) = delete;
  void operator=(CFGBlock &) = delete;
  // allow ilist_traits access.
  friend struct llvm::ilist_traits<CFGBlock>;
};

//===----------------------------------------------------------------------===//
// CFGRegion
//===----------------------------------------------------------------------===//
/// A control-flow region. This class accepts control-flow from the parent
/// operation or a CFG operation. Further this class contains a list of all the
/// CFG blocks directly under it.
class CFGRegion : public CFGFlowPoint {
public:
  static bool classof(CFGPoint const *point);

  /// Returns the parent CFG operation.
  CFGOperation *getParent() const;

  /// Returns the entry block of the region.
  CFGBlock *getEntryBlock() {
    return blocks.empty() ? nullptr : &blocks.front();
  }

  /// Returns the associated region with this CFG point.
  Region *getRegion() const { return region; }

  //===--------------------------------------------------------------------===//
  // Block list management
  //===--------------------------------------------------------------------===//

  using BlockListTy = llvm::iplist<CFGBlock>;
  BlockListTy &getBlocks() { return blocks; }

  // Iteration over the blocks in the region.
  using iterator = BlockListTy::iterator;
  using reverse_iterator = BlockListTy::reverse_iterator;

  iterator begin() { return blocks.begin(); }
  iterator end() { return blocks.end(); }
  reverse_iterator rbegin() { return blocks.rbegin(); }
  reverse_iterator rend() { return blocks.rend(); }

  bool empty() { return blocks.empty(); }
  void push_back(CFGBlock *block) { blocks.push_back(block); }
  void push_front(CFGBlock *block) { blocks.push_front(block); }

  CFGBlock &back() { return blocks.back(); }
  CFGBlock &front() { return blocks.front(); }

  /// Return true if this region has exactly one block.
  bool hasOneBlock() { return !empty() && std::next(begin()) == end(); }

  /// getSublistAccess() - Returns pointer to member of region.
  static BlockListTy CFGRegion::*getSublistAccess(CFGBlock *) {
    return &CFGRegion::blocks;
  }

private:
  CFGRegion(CFGOperation *op, Region &region) : region(&region) {
    this->parent = op;
  }
  Region *region;
  BlockListTy blocks;

  CFGRegion(CFGRegion &) = delete;
  void operator=(CFGRegion &) = delete;
  friend class CFGOperation;
};

//===----------------------------------------------------------------------===//
// CFGOperation
//===----------------------------------------------------------------------===//
/// Base class for all control-flow operations. This class owns the regions as
/// well as successor operands.
class CFGOperation
    : public llvm::ilist_node_with_parent<CFGOperation, CFGBlock> {
public:
  ~CFGOperation();
  /// Returns the CFG op.
  Operation *getOp() const { return opIsTerm.getPointer(); }
  /// Returns the parent block.
  CFGBlock *getParent() const;

  /// Returns the regions in the op.
  ArrayRef<CFGRegion> getRegions() const {
    return ArrayRef<CFGRegion>(regions, getOp()->getNumRegions());
  }

  /// Returns the regions in the op.
  MutableArrayRef<CFGRegion> getRegionsMutable() {
    return MutableArrayRef<CFGRegion>(regions, getOp()->getNumRegions());
  }

  /// Returns the ith-region.
  CFGRegion *getRegion(unsigned i) const {
    assert(i < getOp()->getNumRegions() && "invalid access");
    return regions + i;
  }

  //===--------------------------------------------------------------------===//
  // Successors.
  //===--------------------------------------------------------------------===//
  // Indexed successor access.
  unsigned getNumSuccessors() const { return successors.size(); }
  CFGFlowPoint *getSuccessor(unsigned i) const {
    assert(i < successors.size() && "invalid access");
    return successors[i].get();
  }

  // Successor iteration.
  using succ_iterator = CFGSuccessorRange::iterator;

  /// Returns the successor operands.
  MutableArrayRef<CFGOperand> getSuccessorOperands() {
    return MutableArrayRef<CFGOperand>(successors);
  }

  /// Appends a list of successor operands.
  void appendSuccessors(ArrayRef<CFGFlowPoint *> successors);

  /// Returns whether this op is a terminator.
  bool isTerminator() const { return opIsTerm.getInt(); }

protected:
  CFGOperation(Operation *op, bool isTerminator);

  /// The CFG operation.
  llvm::PointerIntPair<Operation *, 1, bool> opIsTerm;
  /// The CFG operation regions.
  CFGRegion *regions;
  /// The CFG operation successors.
  SmallVector<CFGOperand, 0> successors;
  // allow ilist_traits access.
  friend struct llvm::ilist_traits<CFGOperation>;
  /// Sets the parent block.
  void setParent(CFGBlock *block);
};

//===----------------------------------------------------------------------===//
// CFGOp
//===----------------------------------------------------------------------===//
/// A control-flow op. This class forwards control-flow to its regions and
/// accepts control-flow from nested operations.
class CFGOp : public CFGOperation, public CFGFlowPoint {
public:
  friend class CFGOperation;
  CFGOp(Operation *op) : CFGOperation(op, false) {}
  static bool classof(CFGPoint const *point) {
    return isa<CFGBlock *>(point->getParent());
  }
  static bool classof(CFGOperation const *op) { return !op->isTerminator(); }

  /// Prints the CFG in graphviz format.
  void print(llvm::raw_ostream &stream) const;
  void dump() const { print(llvm::outs()); }
};

//===----------------------------------------------------------------------===//
// CFGTerminator
//===----------------------------------------------------------------------===//
/// A control-flow terminator. This class forwards control-flow to flow-points.
class CFGTerminator : public CFGPoint, public CFGOperation {
public:
  friend class CFGOperation;
  CFGTerminator(Operation *op) : CFGOperation(op, true) {}
  static bool classof(CFGPoint const *point) {
    return isa<CFGBlock *>(point->getParent());
  }
  static bool classof(CFGOperation const *op) { return op->isTerminator(); }
};

//===----------------------------------------------------------------------===//
// CFGContext
//===----------------------------------------------------------------------===//
/// A class for storing a map to
class CFGContext {
public:
  CFGContext() = default;
  /// Looks up a block in the context.
  CFGBlock *lookup(Block *block) {
    return cast_if_present<CFGBlock>(irObjToPoint.lookup(block));
  }
  /// Looks up a region in the context.
  CFGRegion *lookup(Region *region) {
    return cast_if_present<CFGRegion>(irObjToPoint.lookup(region));
  }
  /// Looks up an operation in the context.
  CFGOp *lookupOp(Operation *op) {
    return cast_if_present<CFGOp>(irObjToPoint.lookup(op));
  }
  /// Looks up a terminator in the context.
  CFGTerminator *lookupTerminator(Operation *op) {
    return cast_if_present<CFGTerminator>(irObjToPoint.lookup(op));
  }

  /// Inserts a block in the context.
  void insert(CFGBlock *block) {
    irObjToPoint.insert({block->getBlock(), block});
  }
  /// Inserts a region in the context.
  void insert(CFGRegion *region) {
    irObjToPoint.insert({region->getRegion(), region});
  }
  /// Inserts an operation in the context.
  void insert(CFGOp *op) { irObjToPoint.insert({op->getOp(), op}); }
  /// Inserts an terminator in the context.
  void insert(CFGTerminator *op) { irObjToPoint.insert({op->getOp(), op}); }

  /// Returns the terminators that haven't been resolved.
  const DenseSet<std::pair<CFGLabel, CFGTerminatorOpInterface>> &
  getUnresolvedTerminators() const {
    return unresolvedTerminators;
  }

  /// Erases the unresolved terminators in `terminators`.
  void eraseUnresolvedTerminators(
      ArrayRef<std::pair<CFGLabel, CFGTerminatorOpInterface>> terminators) {
    for (std::pair<CFGLabel, CFGTerminatorOpInterface> term : terminators)
      unresolvedTerminators.erase(term);
  }

  /// Adds a range of unresolved labels with a terminator.
  void appendUnresolvedLabels(CFGTerminatorOpInterface op,
                              ArrayRef<CFGLabel> labels) {
    for (CFGLabel label : labels)
      unresolvedTerminators.insert({label, op});
  }

private:
  DenseMap<void *, CFGPoint *> irObjToPoint;
  DenseSet<std::pair<CFGLabel, CFGTerminatorOpInterface>> unresolvedTerminators;
};

/// Builds the CFG of an operation and its descendents.
CFGOp *buildOpCFG(CFGOpInterface op, CFGContext &context);
} // namespace mlir

#endif // MLIR_IR_CONTROLFLOW_H
