//===- PassDocGen.cpp - MLIR pass documentation generator -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// PassDocGen uses the description of passes to generate documentation.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/CppGen/PassDocGen.h"
#include "mlir/TableGen/GenInfo.h"
#include "llvm/TableGen/Record.h"

static mlir::GenRegistration
    genRegister("gen-pass-doc", "Generate pass documentation",
                [](const llvm::RecordKeeper &records, llvm::raw_ostream &os) {
                  mlir::tblgen::emitPassDocs(records, os);
                  return false;
                });
