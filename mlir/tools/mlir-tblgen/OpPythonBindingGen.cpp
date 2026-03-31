//===- OpPythonBindingGen.cpp - Generator of Python API for MLIR Ops ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/CppGen/OpPythonBindingGen.h"
#include "mlir/TableGen/GenInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using llvm::raw_ostream;
using llvm::RecordKeeper;

static llvm::cl::OptionCategory
    clOpPythonBindingCat("Options for -gen-python-op-bindings");

std::string dialectNameStorage;

llvm::cl::opt<std::string, /*ExternalStorage=*/true>
    clDialectName("bind-dialect",
                  llvm::cl::desc("The dialect to run the generator for"),
                  llvm::cl::location(dialectNameStorage),
                  llvm::cl::cat(clOpPythonBindingCat));

static llvm::cl::opt<std::string> clDialectExtensionName(
    "dialect-extension", llvm::cl::desc("The prefix of the dialect extension"),
    llvm::cl::init(""), llvm::cl::cat(clOpPythonBindingCat));

static GenRegistration
    genPythonBindings("gen-python-op-bindings",
                      "Generate Python bindings for MLIR Ops",
                      [](const RecordKeeper &records, raw_ostream &os) {
                        return tblgen::emitPythonOpBindings(
                            records, dialectNameStorage,
                            clDialectExtensionName, os);
                      });
