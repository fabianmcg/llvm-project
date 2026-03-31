//===- BytecodeDialectGen.cpp - Dialect bytecode read/writer gen  ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/CppGen/BytecodeDialectGen.h"
#include "mlir/TableGen/GenInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"

using namespace llvm;

static cl::OptionCategory dialectGenCat("Options for -gen-bytecode");
static cl::opt<std::string>
    selectedBcDialect("bytecode-dialect", cl::desc("The dialect to gen for"),
                      cl::cat(dialectGenCat), cl::CommaSeparated);

static mlir::GenRegistration
    genBCRW("gen-bytecode", "Generate dialect bytecode readers/writers",
            [](const RecordKeeper &records, raw_ostream &os) {
              return mlir::tblgen::emitBytecodeDialect(records,
                                                       selectedBcDialect, os);
            });
