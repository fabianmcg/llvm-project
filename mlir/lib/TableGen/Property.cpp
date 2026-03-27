//===- Property.cpp - Property wrapper class ----------------------------===//
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

#include "mlir/TableGen/Property.h"
#include "mlir/TableGen/Operator.h"
#include "mlir/TableGen/Predicate.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using namespace mlir::tblgen;

using llvm::DefInit;
using llvm::Init;
using llvm::Record;
using llvm::StringInit;

// Returns the initializer's value as string if the given TableGen initializer
// is a code or string initializer. Returns the empty StringRef otherwise.
static StringRef getValueAsString(const Init *init) {
  if (const auto *str = dyn_cast<StringInit>(init))
    return str->getValue().trim();
  return {};
}

StringRef PropConstraint::getInterfaceType() const {
  return mlir::ods::Constraint::propInterfaceType;
}

Property::Property(const Record *def)
    : Property(
          def, getValueAsString(def->getValueInit("summary")),
          getValueAsString(def->getValueInit("description")),
          getValueAsString(def->getValueInit("storageType")),
          getValueAsString(def->getValueInit("interfaceType")),
          getValueAsString(def->getValueInit("convertFromStorage")),
          getValueAsString(def->getValueInit("assignToStorage")),
          getValueAsString(def->getValueInit("convertToAttribute")),
          getValueAsString(def->getValueInit("convertFromAttribute")),
          getValueAsString(def->getValueInit("parser")),
          getValueAsString(def->getValueInit("optionalParser")),
          getValueAsString(def->getValueInit("printer")),
          getValueAsString(def->getValueInit("readFromMlirBytecode")),
          getValueAsString(def->getValueInit("writeToMlirBytecode")),
          getValueAsString(def->getValueInit("hashProperty")),
          getValueAsString(def->getValueInit("defaultValue")),
          getValueAsString(def->getValueInit("storageTypeValueOverride"))) {
  assert((def->isSubClassOf("Property") || def->isSubClassOf("Attr")) &&
         "must be subclass of TableGen 'Property' class");
}

Property::Property(const DefInit *init) : Property(init->getDef()) {}

Property::Property(const llvm::Record *maybeDef, StringRef summaryArg,
                   StringRef descriptionArg, StringRef storageTypeArg,
                   StringRef interfaceTypeArg,
                   StringRef convertFromStorageCallArg,
                   StringRef assignToStorageCallArg,
                   StringRef convertToAttributeCallArg,
                   StringRef convertFromAttributeCallArg, StringRef parserCallArg,
                   StringRef optionalParserCallArg, StringRef printerCallArg,
                   StringRef readFromMlirBytecodeCallArg,
                   StringRef writeToMlirBytecodeCallArg,
                   StringRef hashPropertyCallArg, StringRef defaultValueArg,
                   StringRef storageTypeValueOverrideArg)
    : PropConstraint(maybeDef, Constraint::CK_Prop) {
  // Populate ods::Property fields.
  storageType =
      storageTypeArg.empty() ? "Property" : storageTypeArg.str();
  interfaceType = interfaceTypeArg.str();
  convertFromStorageCall = convertFromStorageCallArg.str();
  assignToStorageCall = assignToStorageCallArg.str();
  convertToAttributeCall = convertToAttributeCallArg.str();
  convertFromAttributeCall = convertFromAttributeCallArg.str();
  parserCall = parserCallArg.str();
  optionalParserCall = optionalParserCallArg.str();
  printerCall = printerCallArg.str();
  readFromMlirBytecodeCall = readFromMlirBytecodeCallArg.str();
  writeToMlirBytecodeCall = writeToMlirBytecodeCallArg.str();
  hashPropertyCall = hashPropertyCallArg.str();
  defaultValue = defaultValueArg.str();
  storageTypeValueOverride = storageTypeValueOverrideArg.str();

  // For null-def (hardcoded) properties, ods::Constraint::populate() is not
  // called, so manually set summary and description in the ods::Constraint base.
  if (!maybeDef) {
    summary = summaryArg.str();
    description = descriptionArg.str();
  }
}

StringRef Property::getPropertyDefName() const {
  if (def->isAnonymous())
    return getBaseProperty().def->getName();
  return def->getName();
}

Pred Property::getPredicate() const {
  if (!def)
    return Pred();
  const llvm::RecordVal *maybePred = def->getValue("predicate");
  if (!maybePred || !maybePred->getValue())
    return Pred();
  return tblgen::predFromInit(maybePred->getValue());
}

Property Property::getBaseProperty() const {
  if (const auto *defInit =
          llvm::dyn_cast<llvm::DefInit>(def->getValueInit("baseProperty")))
    return Property(defInit).getBaseProperty();
  return *this;
}

bool Property::isSubClassOf(StringRef className) const {
  return def && def->isSubClassOf(className);
}

StringRef ConstantProp::getValue() const {
  return def->getValueAsString("value");
}
