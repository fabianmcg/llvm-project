//===- EnumPythonBindingGen.cpp - Generator of Python API for ODS enums ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/CppGen/EnumPythonBindingGen.h"
#include "mlir/TableGen/GenInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using llvm::raw_ostream;
using llvm::RecordKeeper;

// dialectNameStorage is shared with OpPythonBindingGen.cpp via extern.
extern std::string dialectNameStorage;

static mlir::GenRegistration
    genPythonEnumBindings("gen-python-enum-bindings",
                          "Generate Python bindings for enum attributes",
                          [](const RecordKeeper &records, raw_ostream &os) {
                            return tblgen::emitPythonEnums(records,
                                                           dialectNameStorage,
                                                           os);
                          });
