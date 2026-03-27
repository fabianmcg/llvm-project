//===- Pass.cpp - Pass related classes ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Pass.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using namespace mlir::tblgen;

ods::Pass tblgen::passFromRecord(const llvm::Record *def) {
  ods::Pass pass;
  pass.name = def->getName().str();
  pass.argument = def->getValueAsString("argument").str();
  pass.baseClass = def->getValueAsString("baseClass").str();
  pass.summary = def->getValueAsString("summary").str();
  pass.description = def->getValueAsString("description").str();
  pass.constructor = def->getValueAsString("constructor").str();

  for (const llvm::Record *opt : def->getValueAsListOfDefs("options")) {
    StringRef defaultVal = opt->getValueAsString("defaultValue");
    StringRef flags = opt->getValueAsString("additionalOptFlags");
    pass.options.emplace_back(
        opt->getValueAsString("cppName"), opt->getValueAsString("argument"),
        opt->getValueAsString("type"),
        defaultVal.empty() ? std::optional<StringRef>() : defaultVal,
        opt->getValueAsString("description"),
        flags.empty() ? std::optional<StringRef>() : flags,
        opt->isSubClassOf("ListOption"));
  }

  for (const llvm::Record *stat : def->getValueAsListOfDefs("statistics"))
    pass.statistics.emplace_back(stat->getValueAsString("cppName"),
                                 stat->getValueAsString("name"),
                                 stat->getValueAsString("description"));

  for (StringRef dialect : def->getValueAsListOfStrings("dependentDialects"))
    pass.dependentDialects.push_back(dialect.str());

  return pass;
}
