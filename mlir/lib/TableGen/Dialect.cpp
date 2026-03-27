//===- Dialect.cpp - Dialect wrapper class --------------------------------===//
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

Dialect::Dialect(const llvm::Record *def) : def(def) {
  if (!def)
    return;

  defined = true;

  // Populate string fields.
  name = std::string(def->getValueAsString("name"));
  cppNamespace = std::string(def->getValueAsString("cppNamespace"));

  // cppClassName is derived from the record name by removing underscores.
  cppClassName = def->getName().str();
  llvm::erase(cppClassName, '_');

  summary = std::string(getAsStringOrEmpty(*def, "summary"));
  description = std::string(getAsStringOrEmpty(*def, "description"));

  // Populate dependent dialects.
  for (StringRef d : def->getValueAsListOfStrings("dependentDialects"))
    dependentDialects.push_back(d.str());

  // Populate extra class declaration.
  StringRef extraDecl = def->getValueAsString("extraClassDeclaration");
  if (!extraDecl.empty())
    extraClassDeclaration = extraDecl.str();

  // Populate boolean flags.
  canonicalizer = def->getValueAsBit("hasCanonicalizer");
  constantMaterializer = def->getValueAsBit("hasConstantMaterializer");
  nonDefaultDestructor = def->getValueAsBit("hasNonDefaultDestructor");
  operationAttrVerify = def->getValueAsBit("hasOperationAttrVerify");
  regionArgAttrVerify = def->getValueAsBit("hasRegionArgAttrVerify");
  regionResultAttrVerify = def->getValueAsBit("hasRegionResultAttrVerify");
  operationInterfaceFallback =
      def->getValueAsBit("hasOperationInterfaceFallback");
  defaultAttributePrinterParser =
      def->getValueAsBit("useDefaultAttributePrinterParser");
  defaultTypePrinterParser =
      def->getValueAsBit("useDefaultTypePrinterParser");
  extensible = def->getValueAsBit("isExtensible");
}

const llvm::DagInit *Dialect::getDiscardableAttributes() const {
  return def->getValueAsDag("discardableAttrs");
}
