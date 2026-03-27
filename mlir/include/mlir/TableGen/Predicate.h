//===- Predicate.h - TableGen predicate free functions ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Free functions for constructing mlir::ods::Pred from llvm::Record / Init.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_PREDICATE_H_
#define MLIR_TABLEGEN_PREDICATE_H_

#include "mlir/ODS/Predicate.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/Hashing.h"

namespace llvm {
class Init;
class Record;
class SMLoc;
} // namespace llvm

namespace mlir {
namespace tblgen {

// Re-export the ODS type so existing callers using tblgen::Pred continue to
// work unchanged.
using Pred = mlir::ods::Pred;

/// Constructs an ods::Pred from a TableGen Record, eagerly computing the
/// C++ condition string. Returns the null predicate if \p record is nullptr.
ods::Pred predFromRecord(const llvm::Record *record);

/// Constructs an ods::Pred from a TableGen Init (must be a DefInit or null).
/// Returns the null predicate if \p init does not refer to a Pred def.
ods::Pred predFromInit(const llvm::Init *init);

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_PREDICATE_H_
