//===- Attribute.h - Attribute wrapper class --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Attribute wrapper to simplify using TableGen Record defining a MLIR
// Attribute.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_ATTRIBUTE_H_
#define MLIR_TABLEGEN_ATTRIBUTE_H_

#include "mlir/ODS/Attribute.h"
#include "mlir/Support/LLVM.h"
#include "mlir/TableGen/Constraint.h"

namespace llvm {
class DefInit;
class Record;
} // namespace llvm

namespace mlir {
namespace tblgen {
class Dialect;
class Type;

// Wrapper class with helper methods for accessing attribute constraints defined
// in TableGen.
class AttrConstraint : public Constraint {
public:
  using Constraint::Constraint;

  static bool classof(const Constraint *c) { return c->getKind() == CK_Attr; }

  // Returns true if this constraint is a subclass of the given `className`
  // class defined in TableGen.
  bool isSubClassOf(StringRef className) const;
};

// Wrapper class providing helper methods for accessing MLIR Attribute defined
// in TableGen. This class should closely reflect what is defined as class
// `Attr` in TableGen.
//
// Derives from both tblgen::AttrConstraint (for constraint/predicate access)
// and mlir::ods::Attribute (for eagerly-cached scalar fields). The two base
// classes share no common base, so there is no diamond inheritance.
class Attribute : public AttrConstraint, public mlir::ods::Attribute {
public:
  explicit Attribute(const llvm::Record *record);
  explicit Attribute(const llvm::DefInit *init);

  // Returns the storage type if set. Returns the default storage type
  // ("::mlir::Attribute") otherwise.
  StringRef getStorageType() const {
    return mlir::ods::Attribute::getStorageType();
  }

  // Returns the return type for this attribute.
  StringRef getReturnType() const {
    return mlir::ods::Attribute::getReturnType();
  }

  // Return the type constraint corresponding to the type of this attribute, or
  // std::nullopt if this is not a TypedAttr.
  std::optional<Type> getValueType() const;

  // Returns the template getter method call which reads this attribute's
  // storage and returns the value as of the desired return type.
  // The call will contain a {0} which will be expanded to this attribute.
  StringRef getConvertFromStorageCall() const {
    return mlir::ods::Attribute::getConvertFromStorageCall();
  }

  // Returns true if this attribute can be built from a constant value.
  bool isConstBuildable() const {
    return mlir::ods::Attribute::isConstBuildable();
  }

  // Returns the template that can be used to produce an instance of the
  // attribute. Syntax: `$builder` → a builder, `$0` → the constant value.
  StringRef getConstBuilderTemplate() const {
    return mlir::ods::Attribute::getConstBuilderTemplate();
  }

  // Returns the base-level attribute that this attribute constraint is built
  // upon. This is TableGen-specific as it requires the record hierarchy.
  Attribute getBaseAttr() const;

  // Returns whether this attribute has a default value.
  bool hasDefaultValue() const {
    return mlir::ods::Attribute::hasDefaultValue();
  }

  // Returns the default value for this attribute.
  StringRef getDefaultValue() const {
    return mlir::ods::Attribute::getDefaultValue();
  }

  // Returns whether this attribute is optional.
  bool isOptional() const { return mlir::ods::Attribute::isOptional(); }

  // Returns true if this attribute is a derived attribute (i.e., a subclass
  // of `DerivedAttr`).
  bool isDerivedAttr() const { return mlir::ods::Attribute::isDerivedAttr(); }

  // Returns true if this attribute is a type attribute (i.e., a subclass
  // of `TypeAttrBase`).
  bool isTypeAttr() const { return mlir::ods::Attribute::isTypeAttr(); }

  // Returns true if this attribute is a symbol reference attribute (i.e., a
  // subclass of `SymbolRefAttr` or `FlatSymbolRefAttr`).
  bool isSymbolRefAttr() const {
    return mlir::ods::Attribute::isSymbolRefAttr();
  }

  // Returns true if this attribute is an enum attribute (i.e., a subclass of
  // `EnumAttrInfo`)
  bool isEnumAttr() const { return mlir::ods::Attribute::isEnumAttr(); }

  // Returns this attribute's TableGen def name. If this is an `OptionalAttr`
  // or `DefaultValuedAttr` without explicit name, returns the base attribute's
  // name.
  StringRef getAttrDefName() const {
    return mlir::ods::Attribute::getAttrDefName();
  }

  // Returns the code body for derived attribute. Aborts if this is not a
  // derived attribute.
  StringRef getDerivedCodeBody() const {
    return mlir::ods::Attribute::getDerivedCodeBody();
  }

  // Returns the dialect for the attribute if defined. Returns a tblgen::Dialect
  // wrapping the underlying record.
  Dialect getDialect() const;

  // Returns the TableGen definition this Attribute was constructed from.
  const llvm::Record &getDef() const;
};

// Wrapper class providing helper methods for accessing MLIR constant attribute
// defined in TableGen. This class should closely reflect what is defined as
// class `ConstantAttr` in TableGen.
class ConstantAttr {
public:
  explicit ConstantAttr(const llvm::DefInit *init);

  // Returns the attribute kind.
  Attribute getAttribute() const;

  // Returns the constant value.
  StringRef getConstantValue() const;

private:
  // The TableGen definition of this constant attribute.
  const llvm::Record *def;
};

// Name of infer type op interface.
extern const char *inferTypeOpInterface;

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_ATTRIBUTE_H_
