//===- EnumInfo.cpp - EnumInfo wrapper class ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/EnumInfo.h"
#include "mlir/TableGen/Attribute.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using namespace mlir::tblgen;

using llvm::DefInit;
using llvm::Init;
using llvm::Record;

EnumCase::EnumCase(const Record *record) : def(record) {
  assert(def->isSubClassOf("EnumCase") &&
         "must be subclass of TableGen 'EnumCase' class");
  // Populate ods::EnumCase fields eagerly.
  symbol = def->getValueAsString("symbol").str();
  str = def->getValueAsString("str").str();
  value = def->getValueAsInt("value");
}

EnumCase::EnumCase(const DefInit *init) : EnumCase(init->getDef()) {}

const Record &EnumCase::getDef() const { return *def; }

EnumInfo::EnumInfo(const Record *record) : def(record) {
  assert(isSubClassOf("EnumInfo") &&
         "must be subclass of TableGen 'EnumInfo' class");

  // Populate ods::EnumInfo fields eagerly.
  enumAttr = def->isSubClassOf("EnumAttrInfo");
  bitEnum = def->isSubClassOf("BitEnumBase");
  enumClassName = def->getValueAsString("className").str();
  cppNamespace = def->getValueAsString("cppNamespace").str();
  summary = def->getValueAsString("summary").str();
  description = def->getValueAsString("description").str();
  bitwidth = def->getValueAsInt("bitwidth");
  underlyingType = def->getValueAsString("underlyingType").str();
  underlyingToSymbolFnName =
      def->getValueAsString("underlyingToSymbolFnName").str();
  stringToSymbolFnName = def->getValueAsString("stringToSymbolFnName").str();
  symbolToStringFnName = def->getValueAsString("symbolToStringFnName").str();
  symbolToStringFnRetType =
      def->getValueAsString("symbolToStringFnRetType").str();
  maxEnumValFnName = def->getValueAsString("maxEnumValFnName").str();

  specializedAttr =
      enumAttr && def->getValueAsBit("genSpecializedAttr");
  if (enumAttr)
    specializedAttrClassName =
        def->getValueAsString("specializedAttrClassName").str();
  bitEnumPrimaryGroups =
      bitEnum && def->getValueAsBit("printBitEnumPrimaryGroups");
  bitEnumQuoted = bitEnum && def->getValueAsBit("printBitEnumQuoted");

  // Cache cases as ods::EnumCase objects.
  const auto *inits = def->getValueAsListInit("enumerants");
  cases.reserve(inits->size());
  for (const Init *init : *inits)
    cases.emplace_back(llvm::cast<DefInit>(init)->getDef()->getValueAsString("symbol"),
                       llvm::cast<DefInit>(init)->getDef()->getValueAsString("str"),
                       llvm::cast<DefInit>(init)->getDef()->getValueAsInt("value"));
}

EnumInfo::EnumInfo(const Record &record) : EnumInfo(&record) {}

EnumInfo::EnumInfo(const DefInit *init) : EnumInfo(init->getDef()) {}

bool EnumInfo::isSubClassOf(StringRef className) const {
  return def->isSubClassOf(className);
}

std::optional<Attribute> EnumInfo::asEnumAttr() const {
  if (isEnumAttr())
    return Attribute(def);
  return std::nullopt;
}

std::vector<EnumCase> EnumInfo::getAllCases() const {
  const auto *inits = def->getValueAsListInit("enumerants");

  std::vector<EnumCase> enumCases;
  enumCases.reserve(inits->size());

  for (const Init *init : *inits)
    enumCases.emplace_back(llvm::cast<DefInit>(init));

  return enumCases;
}

const Record *EnumInfo::getBaseAttrClass() const {
  return def->getValueAsDef("baseAttrClass");
}

const Record &EnumInfo::getDef() const { return *def; }
