//===- Attribute.h - ODS Attribute model ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pure C++ model class for MLIR ODS Attribute definitions. No dependency on
// LLVM TableGen. The mlir::tblgen::Attribute subclass populates the fields
// eagerly from an llvm::Record during construction.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ODS_ATTRIBUTE_H_
#define MLIR_ODS_ATTRIBUTE_H_

#include "mlir/ODS/Dialect.h"
#include "mlir/Support/LLVM.h"

#include <optional>
#include <string>

namespace mlir {
namespace ods {

/// Pure C++ model for an MLIR ODS Attribute definition. This class is
/// standalone (does not inherit from ods::Constraint) to avoid diamond
/// inheritance when mlir::tblgen::Attribute derives from both
/// tblgen::AttrConstraint (which inherits ods::Constraint) and this class.
///
/// Fields are populated eagerly by the tblgen::Attribute constructor and
/// are thereafter immutable.
class Attribute {
public:
  Attribute() = default;

  // Returns the storage type for this attribute.
  StringRef getStorageType() const { return storageType; }

  // Returns the return type for this attribute.
  StringRef getReturnType() const { return returnType; }

  // Returns the template getter call that reads this attribute's storage and
  // returns the value as the return type. Contains a {0} placeholder for the
  // attribute itself.
  StringRef getConvertFromStorageCall() const { return convertFromStorage; }

  // Returns true if this attribute can be built from a constant value.
  bool isConstBuildable() const { return !constBuilderTemplate.empty(); }

  // Returns the template used to produce an instance of the attribute from a
  // constant value. $builder → a builder, $0 → the constant value.
  StringRef getConstBuilderTemplate() const { return constBuilderTemplate; }

  // Returns whether this attribute has a default value.
  bool hasDefaultValue() const { return !defaultValue.empty(); }

  // Returns the default value for this attribute.
  StringRef getDefaultValue() const { return defaultValue; }

  // Returns whether this attribute is optional.
  bool isOptional() const { return optional; }

  // Returns true if this is a derived attribute (subclass of DerivedAttr).
  bool isDerivedAttr() const { return derivedAttr; }

  // Returns true if this is a type attribute (subclass of TypeAttrBase).
  bool isTypeAttr() const { return typeAttr; }

  // Returns true if this is a symbol reference attribute.
  bool isSymbolRefAttr() const { return symbolRefAttr; }

  // Returns true if this is an enum attribute (subclass of EnumAttrInfo).
  bool isEnumAttr() const { return enumAttr; }

  // Returns this attribute's TableGen def name. For anonymous optional or
  // default-valued attrs this returns the underlying base attr's name.
  StringRef getAttrDefName() const { return attrDefName; }

  // Returns the code body for derived attributes. Aborts if not a derived
  // attribute.
  StringRef getDerivedCodeBody() const { return derivedCodeBody; }

  // Returns the dialect for this attribute, if one is defined.
  const ods::Dialect &getDialect() const { return dialect; }

protected:
  std::string storageType{"::mlir::Attribute"};
  std::string returnType;
  std::string convertFromStorage;
  std::string constBuilderTemplate;
  std::string defaultValue;
  std::string attrDefName;
  std::string derivedCodeBody;
  ods::Dialect dialect;
  bool optional{false};
  bool derivedAttr{false};
  bool typeAttr{false};
  bool symbolRefAttr{false};
  bool enumAttr{false};
};

} // namespace ods
} // namespace mlir

#endif // MLIR_ODS_ATTRIBUTE_H_
