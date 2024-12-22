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
#include "mlir/IR/Operation.h"
#include "mlir/IR/UseDefLists.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir {
class CFGPoint;
class CFGPointWithSuccessors;
class CFGFlowPoint;
class CFGOp;
class CFGTerminator;

//===----------------------------------------------------------------------===//
// CFGOperand
//===----------------------------------------------------------------------===//

/// Control-flow operand.
class CFGOperand : public IROperand<CFGOperand, CFGFlowPoint *, CFGPoint> {
public:
  using Base = IROperand<CFGOperand, CFGFlowPoint *, CFGPoint>;

  /// Provide the use list that is attached to the given point.
  static IRObjectWithUseList<CFGOperand, CFGPoint> *
  getUseList(CFGFlowPoint *point);

  /// Return which operand this is in the operand list of the point.
  unsigned getOperandNumber();

private:
  friend class CFGOp;
  friend class CFGTerminator;
  CFGOperand(CFGPoint *owner, CFGFlowPoint *point);
};

//===----------------------------------------------------------------------===//
// Predecessors
//===----------------------------------------------------------------------===//

/// Implement a predecessor iterator for control-flow points.
class CFGPredecessorIterator final
    : public llvm::mapped_iterator<ValueUseIterator<CFGOperand, CFGPoint>,
                                   CFGPoint *(*)(CFGOperand &)> {
  static CFGPoint *unwrap(CFGOperand &value);

public:
  /// Initializes the operand type iterator to the specified operand  iterator
  CFGPredecessorIterator(ValueUseIterator<CFGOperand, CFGPoint> it)
      : llvm::mapped_iterator<ValueUseIterator<CFGOperand, CFGPoint>,
                              CFGPoint *(*)(CFGOperand &)>(it, &unwrap) {}

  explicit CFGPredecessorIterator(CFGOperand *operand)
      : CFGPredecessorIterator(
            ValueUseIterator<CFGOperand, CFGPoint>(operand)) {}

  /// Get the successor number in the predecessor terminator.
  unsigned getSuccessorIndex() const;
};

//===----------------------------------------------------------------------===//
// Successors
//===----------------------------------------------------------------------===//

/// This class implements the successor iterators for control flow points.
class CFGSuccessorRange final
    : public llvm::detail::indexed_accessor_range_base<
          CFGSuccessorRange, CFGOperand *, CFGFlowPoint *, CFGFlowPoint *,
          CFGFlowPoint *> {
public:
  using RangeBaseT::RangeBaseT;
  CFGSuccessorRange();
  CFGSuccessorRange(CFGPointWithSuccessors *point);

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
// Control-flow point with successors
//===----------------------------------------------------------------------===//

/// Control-flow point with successors
class CFGPointWithSuccessors {
public:
  //===--------------------------------------------------------------------===//
  // Successors.
  //===--------------------------------------------------------------------===//
  /// Returns true if this blocks has no successors.
  bool hasNoSuccessors() { return successors.empty(); }

  // Indexed successor access.
  unsigned getNumSuccessors() const { return successors.size(); }
  CFGFlowPoint *getSuccessor(unsigned i) const { return successors[i].get(); }

  // Successor iteration.
  using succ_iterator = CFGSuccessorRange::iterator;

  /// Returns the successor operands.
  MutableArrayRef<CFGOperand> getSuccessorOperands() { return successors; }

protected:
  /// The CFG point successors.
  SmallVector<CFGOperand, 0> successors;
};

//===----------------------------------------------------------------------===//
// Control-flow point
//===----------------------------------------------------------------------===//

/// Control-flow point.
class alignas(8) CFGPoint {
public:
  CFGPoint(const CFGPoint &) = delete;
  /// Control-flow point kind.
  typedef enum { FlowOp, FlowRegion, Terminator } CFGPointKind;

  /// Returns the IR object associated with this control-flow point.
  void *getAsOpaquePointer() const { return irObject.getPointer(); }

  /// Returns the control-flow point kind.
  CFGPointKind getKind() const { return irObject.getInt(); }

  static bool classof(CFGPoint const *) { return true; }

  /// Returns whether this control-flow point can have predecessors.
  bool canHavePredecessors() const { return getKind() != Terminator; }

  /// Returns whether this control-flow point can have successors.
  bool canHaveSuccessors() const { return getKind() != FlowRegion; }

  /// Two control-flow points are equal if they have the same IR object.
  bool operator==(const CFGPoint &other) const {
    return getAsOpaquePointer() == other.getAsOpaquePointer();
  }
  bool operator!=(const CFGPoint &rhs) const { return !(*this == rhs); }

private:
  /// The control-flow object.
  llvm::PointerIntPair<void *, 2, CFGPointKind> irObject;

protected:
  CFGPoint(Operation *op, CFGPointKind kind) : irObject(op, kind) {}
  CFGPoint(Region *region) : irObject(region, FlowRegion) {}
};

// Make CFGPoint hashable.
inline ::llvm::hash_code hash_value(const CFGPoint &arg) {
  return llvm::hash_value(arg.getAsOpaquePointer());
}

//===----------------------------------------------------------------------===//
// Control-flow terminator
//===----------------------------------------------------------------------===//

/// Control-flow terminator point.
class CFGTerminator : public CFGPoint, public CFGPointWithSuccessors {
public:
  CFGTerminator(Operation *terminator) : CFGPoint(terminator, Terminator) {}

  static bool classof(CFGPoint const *point) {
    return point->getKind() == Terminator;
  }

  /// Successor iterators.
  succ_iterator succ_begin() { return getSuccessors().begin(); }

  succ_iterator succ_end() { return getSuccessors().end(); }

  /// Returns the successor operands.
  CFGSuccessorRange getSuccessors() { return this; }

  /// Returns the associated terminator operation.
  Operation *getOp() const {
    return reinterpret_cast<Operation *>(getAsOpaquePointer());
  }

  /// Adds a new successor.
  void pushSuccessor(CFGFlowPoint *point) {
    successors.push_back(CFGOperand(this, point));
  }
};

//===----------------------------------------------------------------------===//
// Control-flow point
//===----------------------------------------------------------------------===//

/// Control-flow point with predecessors.
class CFGFlowPoint : public CFGPoint,
                     public IRObjectWithUseList<CFGOperand, CFGPoint> {
public:
  static bool classof(CFGPoint const *point) {
    return point->getKind() == FlowOp || point->getKind() == FlowRegion;
  }

  //===--------------------------------------------------------------------===//
  // Predecessors.
  //===--------------------------------------------------------------------===//
  /// Return true if this block has no predecessors.
  bool hasNoPredecessors() { return pred_begin() == pred_end(); }

  // Predecessor iteration.
  using pred_iterator = CFGPredecessorIterator;

  pred_iterator pred_begin() {
    return pred_iterator((CFGOperand *)getFirstUse());
  }

  pred_iterator pred_end() { return pred_iterator(nullptr); }

  iterator_range<pred_iterator> getPredecessors() {
    return {pred_begin(), pred_end()};
  }

protected:
  CFGFlowPoint(Operation *op) : CFGPoint(op, FlowOp) {}
  CFGFlowPoint(Region *region) : CFGPoint(region) {}
};

//===----------------------------------------------------------------------===//
// Control-flow region
//===----------------------------------------------------------------------===//

/// Control-flow region.
class CFGRegion : public CFGFlowPoint {
public:
  CFGRegion() = delete;
  ~CFGRegion();
  static bool classof(CFGPoint const *point) {
    return point->getKind() == FlowRegion;
  }

  /// Returns the associated region.
  Region *getRegion() const {
    return reinterpret_cast<Region *>(getAsOpaquePointer());
  }

  /// Returns the parent operation.
  CFGOp *getParent() const { return parent; }

private:
  friend class CFGOp;
  CFGRegion(CFGOp *parent, Region *region)
      : CFGFlowPoint(region), parent(parent) {}
  CFGOp *parent;
};

//===----------------------------------------------------------------------===//
// Control-flow operation
//===----------------------------------------------------------------------===//

/// Control-flow operation.
class CFGOp : public CFGFlowPoint, public CFGPointWithSuccessors {
public:
  CFGOp(Operation *op);
  ~CFGOp();
  static bool classof(CFGPoint const *point) {
    return point->getKind() == FlowOp;
  }

  /// Returns the associated operation.
  Operation *getOp() const {
    return reinterpret_cast<Operation *>(getAsOpaquePointer());
  }

  /// Successor iterators.
  succ_iterator succ_begin() { return getSuccessors().begin(); }

  succ_iterator succ_end() { return getSuccessors().end(); }

  /// Returns the successor operands.
  CFGSuccessorRange getSuccessors() { return this; }

  /// Adds a new successor.
  void pushSuccessor(CFGFlowPoint *point) {
    successors.push_back(CFGOperand(this, point));
  }

  /// Returns the CFG regions owned by this op.
  llvm::MutableArrayRef<CFGRegion> getRegions() const {
    return llvm::MutableArrayRef<CFGRegion>(regions, getOp()->getNumRegions());
  }

private:
  CFGRegion *regions;
};

//===----------------------------------------------------------------------===//
// Control-flow graph context
//===----------------------------------------------------------------------===//

/// Control-flow context.
class CFGContext {
  /// Helper for deleting CFG points.
  struct PointDeleter {
    constexpr PointDeleter() noexcept = default;
    void operator()(CFGPoint *point) const;
  };
  using MapTy = DenseMap<void *, std::unique_ptr<CFGPoint, PointDeleter>>;
  /// Lookup table for CFG points.
  MapTy points;

  /// Unresolved CFG edges.
  SmallVector<std::pair<CFGTerminator *, CFGLabel>> unresolvedEdges;

public:
  /// Inserts a new CFG point.
  std::pair<MapTy::iterator, bool> insert(CFGPoint *point) {
    return points.try_emplace(point->getAsOpaquePointer(), point);
  }

  /// Looks up the CFG point corresponding to `op`, returns `nullptr` if the
  /// point doesn't exist.
  CFGPoint *lookup(Operation *op) const {
    if (auto it = points.find(op); it != points.end())
      return it->second.get();
    return nullptr;
  }

  /// Looks up the CFG region corresponding to `region`, returns `nullptr` if
  /// the point doesn't exist.
  CFGRegion *lookup(Region *region) const;

  /// Returns the unresolved CFG edges.
  ArrayRef<std::pair<CFGTerminator *, CFGLabel>> getUnresolvedCFGEdges() const {
    return unresolvedEdges;
  }

  /// Returns the unresolved CFG edges.
  SmallVector<std::pair<CFGTerminator *, CFGLabel>> &getUnresolvedCFGEdges() {
    return unresolvedEdges;
  }

  /// Inserts a new unresolved edge.
  void pushUnresolved(CFGTerminator *term, CFGLabel label) {
    unresolvedEdges.push_back({term, label});
  }
};
} // namespace mlir

#endif // MLIR_IR_CONTROLFLOW_H
