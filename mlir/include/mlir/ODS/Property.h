//===- Property.h - ODS Property model --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pure C++ model class for MLIR ODS Property definitions. No dependency on
// LLVM TableGen. The mlir::tblgen::Property subclass populates the fields
// eagerly from an llvm::Record during construction.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ODS_PROPERTY_H_
#define MLIR_ODS_PROPERTY_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace mlir {
namespace ods {

/// Pure C++ model for an MLIR ODS Property definition. This class is
/// standalone (does not inherit from ods::Constraint) to avoid diamond
/// inheritance when mlir::tblgen::Property derives from both
/// tblgen::PropConstraint (which inherits ods::Constraint) and this class.
///
/// Fields are populated eagerly by the tblgen::Property constructor and
/// are thereafter immutable. Note: summary and description are NOT stored
/// here; they come from ods::Constraint via the tblgen::PropConstraint base.
class Property {
public:
  Property() = default;

  // Returns the storage type.
  StringRef getStorageType() const { return storageType; }

  // Returns the interface type for this property.
  StringRef getInterfaceType() const { return interfaceType; }

  // Returns the template getter method call which reads this property's
  // storage and returns the value as the interface type.
  StringRef getConvertFromStorageCall() const { return convertFromStorageCall; }

  // Returns the template setter method call which stores the interface-type
  // value into the storage.
  StringRef getAssignToStorageCall() const { return assignToStorageCall; }

  // Returns the conversion call that builds an attribute from the storage.
  StringRef getConvertToAttributeCall() const { return convertToAttributeCall; }

  // Returns the setter call which converts from an attribute into the storage.
  StringRef getConvertFromAttributeCall() const {
    return convertFromAttributeCall;
  }

  // Returns the method call which parses this property from textual MLIR.
  StringRef getParserCall() const { return parserCall; }

  // Returns true if this property has defined an optional parser.
  bool hasOptionalParser() const { return !optionalParserCall.empty(); }

  // Returns the method call which optionally parses this property.
  StringRef getOptionalParserCall() const { return optionalParserCall; }

  // Returns the method call which prints this property to textual MLIR.
  StringRef getPrinterCall() const { return printerCall; }

  // Returns the method call which reads this property from bytecode.
  StringRef getReadFromMlirBytecodeCall() const {
    return readFromMlirBytecodeCall;
  }

  // Returns the method call which writes this property to bytecode.
  StringRef getWriteToMlirBytecodeCall() const {
    return writeToMlirBytecodeCall;
  }

  // Returns the code to compute the hash for this property.
  StringRef getHashPropertyCall() const { return hashPropertyCall; }

  // Returns whether this Property has a default value.
  bool hasDefaultValue() const { return !defaultValue.empty(); }

  // Returns the default value for this Property.
  StringRef getDefaultValue() const { return defaultValue; }

  // Returns whether this Property has a distinct storage-type default value.
  bool hasStorageTypeValueOverride() const {
    return !storageTypeValueOverride.empty();
  }

  // Returns the storage-type default value override.
  StringRef getStorageTypeValueOverride() const {
    return storageTypeValueOverride;
  }

protected:
  std::string storageType{"Property"};
  std::string interfaceType;
  std::string convertFromStorageCall;
  std::string assignToStorageCall;
  std::string convertToAttributeCall;
  std::string convertFromAttributeCall;
  std::string parserCall;
  std::string optionalParserCall;
  std::string printerCall;
  std::string readFromMlirBytecodeCall;
  std::string writeToMlirBytecodeCall;
  std::string hashPropertyCall;
  std::string defaultValue;
  std::string storageTypeValueOverride;
};

} // namespace ods
} // namespace mlir

#endif // MLIR_ODS_PROPERTY_H_
