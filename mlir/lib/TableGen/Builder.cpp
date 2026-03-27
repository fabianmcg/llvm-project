//===- Builder.cpp - Builder definitions ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Builder.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using namespace mlir::tblgen;
using llvm::DagInit;
using llvm::DefInit;
using llvm::Init;
using llvm::Record;
using llvm::StringInit;

/// Returns the C++ type string for a builder parameter whose TableGen
/// definition is either a StringInit (the type string directly) or a CArg
/// DefInit (a record with a "type" field).
static StringRef getCppTypeFromInit(const Init *def) {
  if (const auto *stringInit = dyn_cast<StringInit>(def))
    return stringInit->getValue();
  const Record *record = cast<DefInit>(def)->getDef();
  const llvm::RecordVal *type = record->getValue("type");
  if (!type || !type->getValue())
    llvm::PrintFatalError("Builder DAG arguments must be either strings or "
                          "defs which inherit from CArg");
  return record->getValueAsString("type");
}

/// Returns the default value for a builder parameter, or std::nullopt if the
/// parameter has no default. Only CArg DefInits carry default values.
static std::optional<StringRef> getDefaultValueFromInit(const Init *def) {
  if (isa<StringInit>(def))
    return std::nullopt;
  const Record *record = cast<DefInit>(def)->getDef();
  std::optional<StringRef> value =
      record->getValueAsOptionalString("defaultValue");
  return value && !value->empty() ? value : std::nullopt;
}

Builder::Builder(const Record *record, ArrayRef<SMLoc> loc) : def(record) {
  // Populate parameters from the "dagParams" DAG field.
  const DagInit *dag = def->getValueAsDag("dagParams");
  auto *defInit = dyn_cast<DefInit>(dag->getOperator());
  if (!defInit || defInit->getDef()->getName() != "ins")
    PrintFatalError(def->getLoc(), "expected 'ins' in builders");

  bool seenDefaultValue = false;
  for (unsigned i = 0, e = dag->getNumArgs(); i < e; ++i) {
    const StringInit *paramName = dag->getArgName(i);
    const Init *paramValue = dag->getArg(i);

    std::optional<StringRef> name =
        paramName ? std::optional<StringRef>(paramName->getValue())
                  : std::nullopt;
    StringRef cppType = getCppTypeFromInit(paramValue);
    std::optional<StringRef> defaultValue = getDefaultValueFromInit(paramValue);

    Parameter param(name, cppType, defaultValue);

    // Similarly to C++, once an argument with a default value is detected, the
    // following arguments must have default values as well.
    if (param.getDefaultValue()) {
      seenDefaultValue = true;
    } else if (seenDefaultValue) {
      PrintFatalError(loc,
                      "expected an argument with default value after other "
                      "arguments with default values");
    }
    parameters.emplace_back(param);
  }

  // Populate the body field.
  std::optional<StringRef> bodyStr = def->getValueAsOptionalString("body");
  if (bodyStr && !bodyStr->empty())
    body = bodyStr->str();

  // Populate the deprecated message field.
  std::optional<StringRef> msg =
      def->getValueAsOptionalString("odsCppDeprecated");
  if (msg && !msg->empty())
    deprecatedMessage = msg->str();
}
