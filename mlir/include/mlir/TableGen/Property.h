//===- Property.h - Property wrapper class --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Property wrapper to simplify using TableGen Record defining a MLIR
// Property.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_PROPERTY_H_
#define MLIR_TABLEGEN_PROPERTY_H_

#include "mlir/ODS/Property.h"
#include "mlir/Support/LLVM.h"
#include "mlir/TableGen/Constraint.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
class DefInit;
class Record;
} // namespace llvm

namespace mlir {
namespace tblgen {
class Type;
class Pred;

// Wrapper class providing helper methods for accessing property constraint
// values.
class PropConstraint : public Constraint {
public:
  using Constraint::Constraint;

  static bool classof(const Constraint *c) { return c->getKind() == CK_Prop; }

  // Returns the interface type for this constraint. Reads from the TableGen
  // record.
  StringRef getInterfaceType() const;
};

// Wrapper class providing helper methods for accessing MLIR Property defined
// in TableGen. This class should closely reflect what is defined as class
// `Property` in TableGen.
//
// Derives from both tblgen::PropConstraint (for constraint/predicate access)
// and mlir::ods::Property (for eagerly-cached scalar fields). The two base
// classes share no common base, so there is no diamond inheritance.
//
// summary and description are NOT stored here; they are in ods::Constraint
// (accessible via PropConstraint → Constraint → ods::Constraint).
class Property : public PropConstraint, public mlir::ods::Property {
public:
  explicit Property(const llvm::Record *def);
  explicit Property(const llvm::DefInit *init);
  Property(const llvm::Record *maybeDef, StringRef summary,
           StringRef description, StringRef storageType,
           StringRef interfaceType, StringRef convertFromStorageCall,
           StringRef assignToStorageCall, StringRef convertToAttributeCall,
           StringRef convertFromAttributeCall, StringRef parserCall,
           StringRef optionalParserCall, StringRef printerCall,
           StringRef readFromMlirBytecodeCall,
           StringRef writeToMlirBytecodeCall, StringRef hashPropertyCall,
           StringRef defaultValue, StringRef storageTypeValueOverride);

  // Returns the summary (for error messages) of this property's type.
  // Delegates to ods::Constraint::getSummary() which is populated by populate().
  StringRef getSummary() const {
    return mlir::ods::Constraint::getSummary();
  }

  // Returns the description of this property.
  // Delegates to ods::Constraint::getDescription().
  StringRef getDescription() const {
    return mlir::ods::Constraint::getDescription();
  }

  // Returns the storage type.
  StringRef getStorageType() const {
    return mlir::ods::Property::getStorageType();
  }

  // Returns the interface type. Overrides PropConstraint::getInterfaceType()
  // to return the eagerly-cached value.
  StringRef getInterfaceType() const {
    return mlir::ods::Property::getInterfaceType();
  }

  // Returns the template getter method call which reads this property's
  // storage and returns the value as of the desired return type.
  StringRef getConvertFromStorageCall() const {
    return mlir::ods::Property::getConvertFromStorageCall();
  }

  // Returns the template setter method call which reads this property's
  // in the provided interface type and assign it to the storage.
  StringRef getAssignToStorageCall() const {
    return mlir::ods::Property::getAssignToStorageCall();
  }

  // Returns the conversion method call which reads this property's
  // in the storage type and builds an attribute.
  StringRef getConvertToAttributeCall() const {
    return mlir::ods::Property::getConvertToAttributeCall();
  }

  // Returns the setter method call which reads this property's
  // in the provided interface type and assign it to the storage.
  StringRef getConvertFromAttributeCall() const {
    return mlir::ods::Property::getConvertFromAttributeCall();
  }

  // Return the property's predicate. Properties that didn't come from
  // TableGen (the hardcoded ones) have the null predicate.
  Pred getPredicate() const;

  // Returns the method call which parses this property from textual MLIR.
  StringRef getParserCall() const {
    return mlir::ods::Property::getParserCall();
  }

  // Returns true if this property has defined an optional parser.
  bool hasOptionalParser() const {
    return mlir::ods::Property::hasOptionalParser();
  }

  // Returns the method call which optionally parses this property from textual
  // MLIR.
  StringRef getOptionalParserCall() const {
    return mlir::ods::Property::getOptionalParserCall();
  }

  // Returns the method call which prints this property to textual MLIR.
  StringRef getPrinterCall() const {
    return mlir::ods::Property::getPrinterCall();
  }

  // Returns the method call which reads this property from
  // bytecode and assign it to the storage.
  StringRef getReadFromMlirBytecodeCall() const {
    return mlir::ods::Property::getReadFromMlirBytecodeCall();
  }

  // Returns the method call which write this property's
  // to the bytecode.
  StringRef getWriteToMlirBytecodeCall() const {
    return mlir::ods::Property::getWriteToMlirBytecodeCall();
  }

  // Returns the code to compute the hash for this property.
  StringRef getHashPropertyCall() const {
    return mlir::ods::Property::getHashPropertyCall();
  }

  // Returns whether this Property has a default value.
  bool hasDefaultValue() const {
    return mlir::ods::Property::hasDefaultValue();
  }

  // Returns the default value for this Property.
  StringRef getDefaultValue() const {
    return mlir::ods::Property::getDefaultValue();
  }

  // Returns whether this Property has a default storage-type value that is
  // distinct from its default interface-type value.
  bool hasStorageTypeValueOverride() const {
    return mlir::ods::Property::hasStorageTypeValueOverride();
  }

  StringRef getStorageTypeValueOverride() const {
    return mlir::ods::Property::getStorageTypeValueOverride();
  }

  // Returns this property's TableGen def-name.
  StringRef getPropertyDefName() const;

  // Returns the base-level property that this Property constraint is based on
  // or the Property itself otherwise.
  Property getBaseProperty() const;

  // Returns true if this property is backed by a TableGen definition and that
  // definition is a subclass of `className`.
  bool isSubClassOf(StringRef className) const;
};

// A struct wrapping an op property and its name together
struct NamedProperty {
  llvm::StringRef name;
  Property prop;
};

// Wrapper class providing helper methods for processing constant property
// values defined using the `ConstantProp` subclass of `Property`
// in TableGen.
class ConstantProp : public Property {
public:
  explicit ConstantProp(const llvm::DefInit *def) : Property(def) {
    assert(isSubClassOf("ConstantProp"));
  }

  static bool classof(Property *p) { return p->isSubClassOf("ConstantProp"); }

  // Return the constant value of the property as an expression
  // that produces an interface-type constant.
  StringRef getValue() const;
};
} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_PROPERTY_H_
