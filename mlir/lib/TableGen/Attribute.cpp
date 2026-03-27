//===- Attribute.cpp - Attribute wrapper class ----------------------------===//
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

#include "mlir/TableGen/Dialect.h"
#include "mlir/TableGen/Operator.h"
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

bool AttrConstraint::isSubClassOf(StringRef className) const {
  return def->isSubClassOf(className);
}

Attribute::Attribute(const Record *record) : AttrConstraint(record) {
  assert(record->isSubClassOf("Attr") &&
         "must be subclass of TableGen 'Attr' class");

  // Populate ods::Attribute fields eagerly.
  StringRef st = getValueAsString(def->getValueInit("storageType"));
  storageType = st.empty() ? "::mlir::Attribute" : st.str();
  returnType = getValueAsString(def->getValueInit("returnType")).str();
  convertFromStorage =
      getValueAsString(def->getValueInit("convertFromStorage")).str();
  constBuilderTemplate =
      getValueAsString(def->getValueInit("constBuilderCall")).str();
  defaultValue = getValueAsString(def->getValueInit("defaultValue")).str();
  optional = def->getValueAsBit("isOptional");

  derivedAttr = def->isSubClassOf("DerivedAttr");
  typeAttr = def->isSubClassOf("TypeAttrBase");
  enumAttr = def->isSubClassOf("EnumAttrInfo");

  StringRef name = def->getName();
  symbolRefAttr = name == "SymbolRefAttr" || name == "FlatSymbolRefAttr" ||
                  def->isSubClassOf("SymbolRefAttr") ||
                  def->isSubClassOf("FlatSymbolRefAttr");

  // attrDefName: for anonymous attrs use the base attr's name.
  if (def->isAnonymous()) {
    if (const auto *defInit =
            dyn_cast<DefInit>(def->getValueInit("baseAttr"))) {
      // Recurse to get the ultimate base name.
      attrDefName = Attribute(defInit->getDef()).getAttrDefName().str();
    } else {
      attrDefName = name.str();
    }
  } else {
    attrDefName = name.str();
  }

  if (derivedAttr)
    derivedCodeBody = def->getValueAsString("body").str();

  // Populate the ods::Dialect field (slice from tblgen::Dialect to ods::Dialect).
  const llvm::RecordVal *dialectRecord = def->getValue("dialect");
  if (dialectRecord && dialectRecord->getValue()) {
    if (const DefInit *init = dyn_cast<DefInit>(dialectRecord->getValue()))
      dialect = Dialect(init->getDef());
  }
}

Attribute::Attribute(const DefInit *init) : Attribute(init->getDef()) {}

Attribute Attribute::getBaseAttr() const {
  if (const auto *defInit = dyn_cast<DefInit>(def->getValueInit("baseAttr")))
    return Attribute(defInit).getBaseAttr();
  return *this;
}

std::optional<Type> Attribute::getValueType() const {
  if (const auto *defInit = dyn_cast<DefInit>(def->getValueInit("valueType")))
    return Type(defInit->getDef());
  return std::nullopt;
}

Dialect Attribute::getDialect() const {
  const llvm::RecordVal *record = def->getValue("dialect");
  if (record && record->getValue()) {
    if (const DefInit *init = dyn_cast<DefInit>(record->getValue()))
      return Dialect(init->getDef());
  }
  return Dialect(nullptr);
}

const Record &Attribute::getDef() const { return *def; }

ConstantAttr::ConstantAttr(const DefInit *init) : def(init->getDef()) {
  assert(def->isSubClassOf("ConstantAttr") &&
         "must be subclass of TableGen 'ConstantAttr' class");
}

Attribute ConstantAttr::getAttribute() const {
  return Attribute(def->getValueAsDef("attr"));
}

StringRef ConstantAttr::getConstantValue() const {
  return def->getValueAsString("value");
}

const char * ::mlir::tblgen::inferTypeOpInterface = "InferTypeOpInterface";
