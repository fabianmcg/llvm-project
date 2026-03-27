//===- Dialect.cpp - Dialect free function --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Dialect.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using namespace mlir::tblgen;

static StringRef getAsStringOrEmpty(const llvm::Record &record,
                                    StringRef fieldName) {
  if (auto *valueInit = record.getValueInit(fieldName))
    if (llvm::isa<llvm::StringInit>(valueInit))
      return record.getValueAsString(fieldName);
  return "";
}

ods::Dialect tblgen::dialectFromRecord(const llvm::Record *def) {
  ods::Dialect dialect;
  if (!def)
    return dialect;

  dialect.defined = true;

  // Populate string fields.
  dialect.name = std::string(def->getValueAsString("name"));
  dialect.cppNamespace = std::string(def->getValueAsString("cppNamespace"));

  // cppClassName is derived from the record name by removing underscores.
  dialect.cppClassName = def->getName().str();
  llvm::erase(dialect.cppClassName, '_');

  dialect.summary = std::string(getAsStringOrEmpty(*def, "summary"));
  dialect.description = std::string(getAsStringOrEmpty(*def, "description"));

  // Populate dependent dialects.
  for (StringRef d : def->getValueAsListOfStrings("dependentDialects"))
    dialect.dependentDialects.push_back(d.str());

  // Populate extra class declaration.
  StringRef extraDecl = def->getValueAsString("extraClassDeclaration");
  if (!extraDecl.empty())
    dialect.extraClassDeclaration = extraDecl.str();

  // Populate boolean flags.
  dialect.canonicalizer = def->getValueAsBit("hasCanonicalizer");
  dialect.constantMaterializer = def->getValueAsBit("hasConstantMaterializer");
  dialect.nonDefaultDestructor = def->getValueAsBit("hasNonDefaultDestructor");
  dialect.operationAttrVerify = def->getValueAsBit("hasOperationAttrVerify");
  dialect.regionArgAttrVerify = def->getValueAsBit("hasRegionArgAttrVerify");
  dialect.regionResultAttrVerify =
      def->getValueAsBit("hasRegionResultAttrVerify");
  dialect.operationInterfaceFallback =
      def->getValueAsBit("hasOperationInterfaceFallback");
  dialect.defaultAttributePrinterParser =
      def->getValueAsBit("useDefaultAttributePrinterParser");
  dialect.defaultTypePrinterParser =
      def->getValueAsBit("useDefaultTypePrinterParser");
  dialect.extensible = def->getValueAsBit("isExtensible");

  // Populate discardable attributes from the "discardableAttrs" dag field.
  const llvm::DagInit *discardableAttrDag =
      def->getValueAsDag("discardableAttrs");
  for (int i = 0, e = discardableAttrDag->getNumArgs(); i != e; ++i) {
    const llvm::Init *arg = discardableAttrDag->getArg(i);
    StringRef givenName = discardableAttrDag->getArgNameStr(i);
    if (givenName.empty())
      llvm::PrintFatalError(def->getLoc(),
                            "discardable attributes must be named");
    ods::DiscardableAttrInfo info;
    info.name = givenName.str();
    info.type = arg->getAsUnquotedString();
    dialect.discardableAttributes.push_back(std::move(info));
  }

  return dialect;
}
