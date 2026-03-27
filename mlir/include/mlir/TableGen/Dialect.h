//===- Dialect.h - Dialect class --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TableGen wrapper for a MLIR dialect definition. Derives from mlir::ods::Dialect
// and populates the ODS fields from an llvm::Record in its constructor.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_DIALECT_H_
#define MLIR_TABLEGEN_DIALECT_H_

#include "mlir/ODS/Dialect.h"
#include "mlir/Support/LLVM.h"

namespace llvm {
class DagInit;
class Record;
} // namespace llvm

namespace mlir {
namespace tblgen {

/// TableGen wrapper for a MLIR dialect. Derives from mlir::ods::Dialect and
/// additionally stores the underlying llvm::Record for TableGen-specific
/// accessors. Equality uses pointer identity on the record.
class Dialect : public mlir::ods::Dialect {
public:
  explicit Dialect(const llvm::Record *def);

  /// Returns the discardable attribute dag defined in TableGen. This is
  /// TableGen-specific and has no ODS equivalent.
  const llvm::DagInit *getDiscardableAttributes() const;

  const llvm::Record *getDef() const { return def; }

  // Records are pointer-comparable; override the ODS name-based equality.
  bool operator==(const Dialect &other) const { return def == other.def; }
  bool operator!=(const Dialect &other) const { return !(*this == other); }

  // Retain name-based ordering from ODS.
  bool operator<(const Dialect &other) const {
    return getName() < other.getName();
  }

  // Returns whether the dialect is defined.
  explicit operator bool() const { return def != nullptr; }
  bool isDefined() const { return def != nullptr; }

private:
  const llvm::Record *def{nullptr};
};

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_DIALECT_H_
