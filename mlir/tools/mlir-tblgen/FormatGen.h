//===- FormatGen.h - Utilities for custom assembly formats ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains common classes for building custom assembly format parsers
// and generators.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRTBLGEN_FORMATGEN_H_
#define MLIR_TOOLS_MLIRTBLGEN_FORMATGEN_H_

#include "mlir/TableGen/CppGen/FormatGen.h"
#include "llvm/Support/CommandLine.h"

namespace mlir {
namespace tblgen {

/// Whether a failure in parsing the assembly format should be a fatal error.
extern llvm::cl::opt<bool> formatErrorIsFatal;

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TOOLS_MLIRTBLGEN_FORMATGEN_H_
