//===- OpFormatGen.h - MLIR operation format generator ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interface for generating parsers and printers from the
// declarative format.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_CPPGEN_OPFORMATGEN_H
#define MLIR_TABLEGEN_CPPGEN_OPFORMATGEN_H

namespace mlir {
namespace tblgen {
class OpClass;
class Operator;

/// Generate the assembly format for the given operator. If \p fatalOnError is
/// true, format parse errors cause the process to exit; otherwise they are
/// silently ignored.
void generateOpFormat(const Operator &constOp, OpClass &opClass,
                      bool hasProperties, bool fatalOnError = true);

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_CPPGEN_OPFORMATGEN_H
