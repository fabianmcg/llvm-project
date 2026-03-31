//===- EnumsGen.cpp - MLIR enum utility generator -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// EnumsGen generates common utility functions for enums.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/CppGen/EnumsGen.h"
#include "mlir/TableGen/GenInfo.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"

// Registers the enum utility generator to mlir-tblgen.
static mlir::GenRegistration
    genEnumDecls("gen-enum-decls", "Generate enum utility declarations",
                 [](const llvm::RecordKeeper &records, llvm::raw_ostream &os) {
                   return mlir::tblgen::emitEnumDecls(records, os);
                 });

// Registers the enum utility generator to mlir-tblgen.
static mlir::GenRegistration
    genEnumDefs("gen-enum-defs", "Generate enum utility definitions",
                [](const llvm::RecordKeeper &records, llvm::raw_ostream &os) {
                  return mlir::tblgen::emitEnumDefs(records, os);
                });
