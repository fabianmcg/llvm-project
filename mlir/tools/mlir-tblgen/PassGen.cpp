//===- PassGen.cpp - MLIR pass registration generator ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// PassGen uses the description of passes to generate base classes for passes
// and command line registration.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/CppGen/PassGen.h"
#include "mlir/TableGen/GenInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/TableGen/Record.h"

static llvm::cl::OptionCategory passGenCat("Options for -gen-pass-decls");
static llvm::cl::opt<std::string>
    groupName("name", llvm::cl::desc("The name of this group of passes"),
              llvm::cl::cat(passGenCat));

static mlir::GenRegistration
    genPassDecls("gen-pass-decls", "Generate pass declarations",
                 [](const llvm::RecordKeeper &records, llvm::raw_ostream &os) {
                   mlir::tblgen::emitPasses(records, groupName, os);
                   return false;
                 });
