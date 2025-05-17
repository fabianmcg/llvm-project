//===- ControlFlowLabel.h - MLIR CFG Label -----------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares MLIR's control-flow label attributes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_CONTROLFLOWLABEL_H
#define MLIR_IR_CONTROLFLOWLABEL_H

#include "mlir/IR/Attributes.h"
#include "llvm/Support/PointerLikeTypeTraits.h"

namespace mlir {
class CFGLabel;
class WalkResult;

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

#endif // MLIR_IR_CONTROLFLOWLABEL_H
