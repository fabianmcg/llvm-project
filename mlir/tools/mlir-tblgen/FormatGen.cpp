//===- FormatGen.cpp - Utilities for custom assembly formats --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FormatGen.h"

//===----------------------------------------------------------------------===//
// Commandline Options
//===----------------------------------------------------------------------===//

llvm::cl::opt<bool> mlir::tblgen::formatErrorIsFatal(
    "asmformat-error-is-fatal",
    llvm::cl::desc("Emit a fatal error if format parsing fails"),
    llvm::cl::init(true));
