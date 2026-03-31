//===- DialectInterfacesGen.cpp - MLIR dialect interface generator --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/CppGen/DialectInterfacesGen.h"
#include "mlir/TableGen/GenInfo.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using llvm::raw_ostream;
using llvm::RecordKeeper;

static mlir::GenRegistration genDecls(
    "gen-dialect-interface-decls", "Generate dialect interface declarations.",
    [](const RecordKeeper &records, raw_ostream &os) {
      return tblgen::DialectInterfaceGenerator(records, os).emitInterfaceDecls();
    });
