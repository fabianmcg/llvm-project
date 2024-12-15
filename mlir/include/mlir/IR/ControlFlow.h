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

#include "mlir/IR/Attributes.h"
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
class CFGLabel;
class WalkResult;

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
// Control-flow operation
//===----------------------------------------------------------------------===//

/// Control-flow operation.
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
// Control-flow operation
//===----------------------------------------------------------------------===//

/// Control-flow operation.
class CFGOp : public CFGFlowPoint, public CFGPointWithSuccessors {
public:
  CFGOp(Operation *op) : CFGFlowPoint(op) {}
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
};

//===----------------------------------------------------------------------===//
// Control-flow region
//===----------------------------------------------------------------------===//

/// Control-flow region.
class CFGRegion : public CFGFlowPoint {
public:
  CFGRegion(Region *region) : CFGFlowPoint(region) {}

  static bool classof(CFGPoint const *point) {
    return point->getKind() == FlowRegion;
  }

  /// Returns the associated region.
  Region *getRegion() const {
    return reinterpret_cast<Region *>(getAsOpaquePointer());
  }
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

  /// Looks up the CFG point corresponding to `region`, returns `nullptr` if the
  /// point doesn't exist.
  CFGPoint *lookup(Region *region) const {
    if (auto it = points.find(region); it != points.end())
      return it->second.get();
    return nullptr;
  }
};

//===----------------------------------------------------------------------===//
// CFGLabelAttr
//===----------------------------------------------------------------------===//

/// CFGLabel objects represent CFG labels information in MLIR.
/// CFGLabelAttr acts as the anchor for all CFGLabel based attributes.
class CFGLabelAttr : public Attribute {
public:
  using Attribute::Attribute;

  /// Walk all of the labels nested directly under, and including, the
  /// current. This means that if a label is nested under a non-label
  /// attribute, it will *not* be walked by this method. This walk is performed
  /// in pre-order to get this behavior.
  WalkResult walk(function_ref<WalkResult(CFGLabel)> walkFn);

  /// Return an instance of the given label type if one is nested under the
  /// current label. Returns nullptr if one could not be found.
  template <typename T>
  T findInstanceOf() {
    T result = {};
    walk([&](auto loc) {
      if (auto typedLoc = llvm::dyn_cast<T>(loc)) {
        result = typedLoc;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return result;
  }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(Attribute attr);
};

//===----------------------------------------------------------------------===//
// CFGLabel
//===----------------------------------------------------------------------===//

/// This class defines the main interface for cfg labels in MLIR and acts as a
/// non-nullable wrapper around a CFGLabelAttr.
class CFGLabel {
public:
  CFGLabel(CFGLabelAttr loc) : impl(loc) {
    assert(loc && "label should never be null.");
  }
  CFGLabel(const CFGLabelAttr::ImplType *impl) : impl(impl) {
    assert(impl && "label should never be null.");
  }

  /// Return the context this label is uniqued in.
  MLIRContext *getContext() const { return impl.getContext(); }

  /// Access the impl label attribute.
  operator CFGLabelAttr() const { return impl; }
  CFGLabelAttr *operator->() const { return const_cast<CFGLabelAttr *>(&impl); }

  /// Comparison operators.
  bool operator==(CFGLabel rhs) const { return impl == rhs.impl; }
  bool operator!=(CFGLabel rhs) const { return !(*this == rhs); }

  /// Print the label.
  void print(raw_ostream &os) const { impl.print(os); }
  void dump() const { impl.dump(); }

  friend ::llvm::hash_code hash_value(CFGLabel arg);

  /// Methods for supporting PointerLikeTypeTraits.
  const void *getAsOpaquePointer() const { return impl.getAsOpaquePointer(); }
  static CFGLabel getFromOpaquePointer(const void *pointer) {
    return CFGLabelAttr(reinterpret_cast<const AttributeStorage *>(pointer));
  }

  /// Support llvm style casting.
  static bool classof(Attribute attr) { return llvm::isa<CFGLabelAttr>(attr); }

protected:
  /// The internal backing label attribute.
  CFGLabelAttr impl;
};

inline raw_ostream &operator<<(raw_ostream &os, const CFGLabel &loc) {
  loc.print(os);
  return os;
}

// Make CFGLabel hashable.
inline ::llvm::hash_code hash_value(CFGLabel arg) {
  return hash_value(arg.impl);
}

//===----------------------------------------------------------------------===//
// SubElements
//===----------------------------------------------------------------------===//

/// Enable labels to be introspected as sub-elements.
template <>
struct AttrTypeSubElementHandler<CFGLabel> {
  static void walk(CFGLabel param, AttrTypeImmediateSubElementWalker &walker) {
    walker.walk(param);
  }
  static CFGLabel replace(CFGLabel param, AttrSubElementReplacements &attrRepls,
                          TypeSubElementReplacements &typeRepls) {
    return cast<CFGLabelAttr>(attrRepls.take_front(1)[0]);
  }
};
} // namespace mlir

//===----------------------------------------------------------------------===//
// LLVM Utilities
//===----------------------------------------------------------------------===//

namespace llvm {
// Type hash just like pointers.
template <>
struct DenseMapInfo<mlir::CFGLabel> {
  static mlir::CFGLabel getEmptyKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::CFGLabel::getFromOpaquePointer(pointer);
  }
  static mlir::CFGLabel getTombstoneKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::CFGLabel::getFromOpaquePointer(pointer);
  }
  static unsigned getHashValue(mlir::CFGLabel val) {
    return mlir::hash_value(val);
  }
  static bool isEqual(mlir::CFGLabel LHS, mlir::CFGLabel RHS) {
    return LHS == RHS;
  }
};

/// We align CFGLabelStorage by 8, so allow LLVM to steal the low bits.
template <>
struct PointerLikeTypeTraits<mlir::CFGLabel> {
public:
  static inline void *getAsVoidPointer(mlir::CFGLabel I) {
    return const_cast<void *>(I.getAsOpaquePointer());
  }
  static inline mlir::CFGLabel getFromVoidPointer(void *P) {
    return mlir::CFGLabel::getFromOpaquePointer(P);
  }
  static constexpr int NumLowBitsAvailable =
      PointerLikeTypeTraits<mlir::Attribute>::NumLowBitsAvailable;
};

/// The constructors in mlir::CFGLabel ensure that the class is a non-nullable
/// wrapper around mlir::CFGLabelAttr. Override default behavior and always
/// return true for isPresent().
template <>
struct ValueIsPresent<mlir::CFGLabel> {
  using UnwrappedType = mlir::CFGLabel;
  static inline bool isPresent(const mlir::CFGLabel &label) { return true; }
};

/// Add support for llvm style casts. We provide a cast between To and From if
/// From is mlir::CFGLabel or derives from it.
template <typename To, typename From>
struct CastInfo<To, From,
                std::enable_if_t<
                    std::is_same_v<mlir::CFGLabel, std::remove_const_t<From>> ||
                    std::is_base_of_v<mlir::CFGLabel, From>>>
    : DefaultDoCastIfPossible<To, From, CastInfo<To, From>> {

  static inline bool isPossible(mlir::CFGLabel label) {
    /// Return a constant true instead of a dynamic true when casting to self or
    /// up the hierarchy. Additionally, all casting info is deferred to the
    /// wrapped mlir::CFGLabelAttr instance stored in mlir::CFGLabel.
    return std::is_same_v<To, std::remove_const_t<From>> ||
           isa<To>(static_cast<mlir::CFGLabelAttr>(label));
  }

  static inline To castFailed() { return To(); }

  static inline To doCast(mlir::CFGLabel label) { return To(label->getImpl()); }
};
} // namespace llvm

#endif // MLIR_IR_CONTROLFLOW_H
