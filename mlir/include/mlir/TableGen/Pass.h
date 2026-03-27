//===- Pass.h - TableGen pass definitions -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Free function for constructing an mlir::ods::Pass from an llvm::Record.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_PASS_H_
#define MLIR_TABLEGEN_PASS_H_

#include "mlir/ODS/Pass.h"

namespace llvm {
class Record;
} // namespace llvm

namespace mlir {
namespace tblgen {

// Re-export ODS types so callers using tblgen::PassOption / tblgen::PassStatistic
// continue to work unchanged.
using PassOption = mlir::ods::PassOption;
using PassStatistic = mlir::ods::PassStatistic;
using Pass = mlir::ods::Pass;

/// Constructs an ods::Pass by reading all fields from the TableGen record.
ods::Pass passFromRecord(const llvm::Record *def);

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_PASS_H_
