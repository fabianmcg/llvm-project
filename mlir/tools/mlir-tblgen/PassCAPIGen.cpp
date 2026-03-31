//===- PassCAPIGen.cpp - MLIR pass C API generator ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// PassCAPIGen uses the description of passes to generate C API for the passes.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/CppGen/PassCAPIGen.h"
#include "mlir/TableGen/GenInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/TableGen/Record.h"

static llvm::cl::OptionCategory
    passGenCat("Options for -gen-pass-capi-header and -gen-pass-capi-impl");
static llvm::cl::opt<std::string>
    groupName("prefix",
              llvm::cl::desc("The prefix to use for this group of passes. The "
                             "form will be mlirCreate<prefix><passname>, the "
                             "prefix can avoid conflicts across libraries."),
              llvm::cl::cat(passGenCat));

static mlir::GenRegistration
    genCAPIHeader("gen-pass-capi-header", "Generate pass C API header",
                  [](const llvm::RecordKeeper &records, llvm::raw_ostream &os) {
                    mlir::tblgen::emitPassCAPIHeader(records, groupName, os);
                    return false;
                  });

static mlir::GenRegistration
    genCAPIImpl("gen-pass-capi-impl", "Generate pass C API implementation",
                [](const llvm::RecordKeeper &records, llvm::raw_ostream &os) {
                  mlir::tblgen::emitPassCAPIImpl(records, groupName, os);
                  return false;
                });
