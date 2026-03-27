//===- Type.cpp - Type class ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Type.h"
#include "mlir/TableGen/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using namespace mlir::tblgen;
using llvm::Record;

static std::optional<std::string>
computeBuilderCall(const llvm::Record *def, bool isVariableLength) {
  const Record *baseType = def;
  if (isVariableLength)
    baseType = baseType->getValueAsDef("baseType");

  const llvm::RecordVal *builderCall = baseType->getValue("builderCall");
  if (!builderCall || !builderCall->getValue())
    return std::nullopt;
  return llvm::TypeSwitch<const llvm::Init *, std::optional<std::string>>(
             builderCall->getValue())
      .Case([](const llvm::StringInit *init) -> std::optional<std::string> {
        StringRef value = init->getValue();
        return value.empty() ? std::nullopt
                             : std::optional<std::string>(value.str());
      })
      .Default(std::nullopt);
}

TypeConstraint::TypeConstraint(const llvm::Record *record)
    : Constraint(record) {
  optional = def->isSubClassOf("Optional");
  variadicOfVariadic = def->isSubClassOf("VariadicOfVariadic");
  if (variadicOfVariadic)
    segmentSizeAttr =
        def->getValueAsString("segmentAttrName").str();
  builderCall = computeBuilderCall(def, isVariableLength());
  cppType = def->getValueAsString("cppType").str();
}

TypeConstraint::TypeConstraint(const llvm::DefInit *init)
    : TypeConstraint(init->getDef()) {}

Type::Type(const Record *record) : TypeConstraint(record) {}

ods::Dialect Type::getDialect() const {
  return tblgen::dialectFromRecord(def->getValueAsDef("dialect"));
}
