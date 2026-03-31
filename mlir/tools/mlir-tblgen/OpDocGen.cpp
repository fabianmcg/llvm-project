//===- OpDocGen.cpp - MLIR operation documentation generator --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// OpDocGen uses the description of operations to generate documentation for the
// operations.
//
//===----------------------------------------------------------------------===//

#include "DialectGenUtilities.h"
#include "OpGenHelpers.h"
#include "mlir/TableGen/CppGen/OpDocGen.h"
#include "mlir/TableGen/Dialect.h"
#include "mlir/TableGen/GenInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::tblgen;

//===----------------------------------------------------------------------===//
// Commandline Options
//===----------------------------------------------------------------------===//

static cl::OptionCategory
    docCat("Options for -gen-(attrdef|typedef|enum|op|dialect)-doc");
static cl::opt<std::string>
    stripPrefix("strip-prefix",
                cl::desc("Strip prefix of the fully qualified names"),
                cl::init("::mlir::"), cl::cat(docCat));
static cl::opt<bool> allowHugoSpecificFeatures(
    "allow-hugo-specific-features",
    cl::desc("Allows using features specific to Hugo"), cl::init(false),
    cl::cat(docCat));
static cl::opt<bool>
    keepOpSourceOrder("keep-op-source-order",
                      cl::desc("Do not sort ops alphabetically"),
                      cl::init(false), cl::cat(docCat));

//===----------------------------------------------------------------------===//
// Gen Registration
//===----------------------------------------------------------------------===//

/// Collect records and invoke \p fn if a dialect could be determined.
static bool withDialectRecords(
    const RecordKeeper &records,
    llvm::function_ref<bool(const DialectRecords &)> fn) {
  auto dialectDefs = records.getAllDerivedDefinitionsIfDefined("Dialect");
  SmallVector<Dialect> dialects(dialectDefs.begin(), dialectDefs.end());
  std::optional<Dialect> dialect = findDialectToGenerate(dialects);
  if (!dialect)
    return true;
  std::optional<DialectRecords> filtered = collectRecords(
      records, getRequestedOpDefinitions(records), *dialect, keepOpSourceOrder);
  if (!filtered)
    return true;
  return fn(*filtered);
}

static mlir::GenRegistration
    genAttrRegister("gen-attrdef-doc",
                    "Generate dialect attribute documentation",
                    [](const RecordKeeper &records, raw_ostream &os) {
                      return withDialectRecords(records, [&](const DialectRecords &r) {
                        return emitAttrDefDoc(r, os);
                      });
                    });

static mlir::GenRegistration
    genOpRegister("gen-op-doc", "Generate dialect documentation",
                  [](const RecordKeeper &records, raw_ostream &os) {
                    return withDialectRecords(records, [&](const DialectRecords &r) {
                      return emitOpDoc(r, stripPrefix, allowHugoSpecificFeatures,
                                       os);
                    });
                  });

static mlir::GenRegistration
    genTypeRegister("gen-typedef-doc", "Generate dialect type documentation",
                    [](const RecordKeeper &records, raw_ostream &os) {
                      return withDialectRecords(records, [&](const DialectRecords &r) {
                        return emitTypeDefDoc(r, os);
                      });
                    });

static mlir::GenRegistration
    genEnumRegister("gen-enum-doc", "Generate dialect enum documentation",
                    [](const RecordKeeper &records, raw_ostream &os) {
                      return withDialectRecords(records, [&](const DialectRecords &r) {
                        return emitEnumDoc(r, os);
                      });
                    });

static mlir::GenRegistration
    genRegister("gen-dialect-doc", "Generate dialect documentation",
                [](const RecordKeeper &records, raw_ostream &os) {
                  return withDialectRecords(records, [&](const DialectRecords &r) {
                    return emitDialectDoc(r, stripPrefix,
                                          allowHugoSpecificFeatures, os);
                  });
                });
