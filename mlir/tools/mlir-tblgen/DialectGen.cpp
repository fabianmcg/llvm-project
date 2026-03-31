//===- DialectGen.cpp - MLIR dialect definitions generator ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DialectGen uses the description of dialects to generate C++ definitions.
//
//===----------------------------------------------------------------------===//

#include "DialectGenUtilities.h"
#include "mlir/TableGen/CppGen/DialectGen.h"
#include "mlir/TableGen/GenInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/TableGen/Record.h"

static llvm::cl::OptionCategory dialectGenCat("Options for -gen-dialect-*");
static llvm::cl::opt<std::string>
    selectedDialect("dialect", llvm::cl::desc("The dialect to gen for"),
                    llvm::cl::cat(dialectGenCat), llvm::cl::CommaSeparated);

// Tool-level wrapper that reads the CLI option and forwards to the library.
// This function is declared in DialectGenUtilities.h and used by OpDocGen.cpp.
std::optional<mlir::tblgen::Dialect>
mlir::tblgen::findDialectToGenerate(llvm::ArrayRef<Dialect> dialects) {
  return mlir::tblgen::findDialectToGenerate(
      dialects,
      selectedDialect.getNumOccurrences() > 0 ? selectedDialect.getValue()
                                               : "");
}

static mlir::GenRegistration
    genDialectDecls("gen-dialect-decls", "Generate dialect declarations",
                    [](const llvm::RecordKeeper &records,
                       llvm::raw_ostream &os) {
                      return mlir::tblgen::emitDialectDecls(
                          records, selectedDialect, os);
                    });

static mlir::GenRegistration
    genDialectDefs("gen-dialect-defs", "Generate dialect definitions",
                   [](const llvm::RecordKeeper &records,
                      llvm::raw_ostream &os) {
                     return mlir::tblgen::emitDialectDefs(records,
                                                          selectedDialect, os);
                   });
