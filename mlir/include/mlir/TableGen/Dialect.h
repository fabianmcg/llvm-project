//===- Dialect.h - Dialect free functions -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Free function to construct an mlir::ods::Dialect from an llvm::Record.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_DIALECT_H_
#define MLIR_TABLEGEN_DIALECT_H_

#include "mlir/ODS/Dialect.h"

namespace llvm {
class Record;
} // namespace llvm

namespace mlir {
namespace tblgen {

/// Constructs an ods::Dialect by eagerly reading all ODS fields from the
/// TableGen record, including populating discardableAttributes from the
/// "discardableAttrs" dag field. Returns an undefined (default-constructed)
/// ods::Dialect if def is null.
ods::Dialect dialectFromRecord(const llvm::Record *def);

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_DIALECT_H_
