//===- AttrOrTypeDefGen.cpp - MLIR AttrOrType definitions generator -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FormatGen.h"
#include "mlir/TableGen/CppGen/AttrOrTypeDefGen.h"
#include "mlir/TableGen/GenInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using namespace mlir::tblgen;
using llvm::RecordKeeper;
using llvm::raw_ostream;

//===----------------------------------------------------------------------===//
// AttrDef
//===----------------------------------------------------------------------===//

static llvm::cl::OptionCategory attrdefGenCat("Options for -gen-attrdef-*");
static llvm::cl::opt<std::string>
    attrDialect("attrdefs-dialect",
                llvm::cl::desc("Generate attributes for this dialect"),
                llvm::cl::cat(attrdefGenCat), llvm::cl::CommaSeparated);

static mlir::GenRegistration
    genAttrDefs("gen-attrdef-defs", "Generate AttrDef definitions",
                [](const RecordKeeper &records, raw_ostream &os) {
                  AttrDefGenerator generator(records, os, formatErrorIsFatal);
                  return generator.emitDefs(attrDialect);
                });
static mlir::GenRegistration
    genAttrDecls("gen-attrdef-decls", "Generate AttrDef declarations",
                 [](const RecordKeeper &records, raw_ostream &os) {
                   AttrDefGenerator generator(records, os, formatErrorIsFatal);
                   return generator.emitDecls(attrDialect);
                 });

static mlir::GenRegistration
    genAttrConstrDefs("gen-attr-constraint-defs",
                      "Generate attribute constraint definitions",
                      [](const RecordKeeper &records, raw_ostream &os) {
                        emitAttrConstraintDefs(records, os);
                        return false;
                      });
static mlir::GenRegistration
    genAttrConstrDecls("gen-attr-constraint-decls",
                       "Generate attribute constraint declarations",
                       [](const RecordKeeper &records, raw_ostream &os) {
                         emitAttrConstraintDecls(records, os);
                         return false;
                       });

//===----------------------------------------------------------------------===//
// TypeDef
//===----------------------------------------------------------------------===//

static llvm::cl::OptionCategory typedefGenCat("Options for -gen-typedef-*");
static llvm::cl::opt<std::string>
    typeDialect("typedefs-dialect",
                llvm::cl::desc("Generate types for this dialect"),
                llvm::cl::cat(typedefGenCat), llvm::cl::CommaSeparated);

static mlir::GenRegistration
    genTypeDefs("gen-typedef-defs", "Generate TypeDef definitions",
                [](const RecordKeeper &records, raw_ostream &os) {
                  TypeDefGenerator generator(records, os, formatErrorIsFatal);
                  return generator.emitDefs(typeDialect);
                });
static mlir::GenRegistration
    genTypeDecls("gen-typedef-decls", "Generate TypeDef declarations",
                 [](const RecordKeeper &records, raw_ostream &os) {
                   TypeDefGenerator generator(records, os, formatErrorIsFatal);
                   return generator.emitDecls(typeDialect);
                 });

static mlir::GenRegistration
    genTypeConstrDefs("gen-type-constraint-defs",
                      "Generate type constraint definitions",
                      [](const RecordKeeper &records, raw_ostream &os) {
                        emitTypeConstraintDefs(records, os);
                        return false;
                      });
static mlir::GenRegistration
    genTypeConstrDecls("gen-type-constraint-decls",
                       "Generate type constraint declarations",
                       [](const RecordKeeper &records, raw_ostream &os) {
                         emitTypeConstraintDecls(records, os);
                         return false;
                       });
