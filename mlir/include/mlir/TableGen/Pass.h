//===- Pass.h - TableGen pass definitions -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TableGen wrapper around ODS Pass classes. Derives from mlir::ods::Pass and
// populates all ODS fields from an llvm::Record in its constructor.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_PASS_H_
#define MLIR_TABLEGEN_PASS_H_

#include "mlir/ODS/Pass.h"
#include "mlir/Support/LLVM.h"

namespace llvm {
class Record;
} // namespace llvm

namespace mlir {
namespace tblgen {

// Re-export ODS types so callers using tblgen::PassOption / tblgen::PassStatistic
// continue to work unchanged.
using PassOption = mlir::ods::PassOption;
using PassStatistic = mlir::ods::PassStatistic;

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

/// TableGen wrapper for a pass definition. Derives from mlir::ods::Pass and
/// additionally stores the underlying llvm::Record.
class Pass : public mlir::ods::Pass {
public:
  explicit Pass(const llvm::Record *def);

  const llvm::Record *getDef() const { return def; }

private:
  const llvm::Record *def;
};

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_PASS_H_
