//===- Constraint.h - Constraint class --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TableGen wrapper around ODS Constraint. Derives from mlir::ods::Constraint
// and populates all ODS fields from an llvm::Record in its constructor.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_CONSTRAINT_H_
#define MLIR_TABLEGEN_CONSTRAINT_H_

#include "mlir/ODS/Constraint.h"
#include "mlir/Support/LLVM.h"
#include "mlir/TableGen/Predicate.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
class Record;
} // namespace llvm

namespace mlir {
namespace tblgen {

/// TableGen wrapper for a constraint. Derives from mlir::ods::Constraint and
/// additionally stores the underlying llvm::Record. Equality uses pointer
/// identity on the record. ODS fields are populated eagerly in the
/// constructor.
class Constraint : public mlir::ods::Constraint {
public:
  // Re-export Kind so callers using Constraint::CK_* continue to work.
  using Kind = mlir::ods::Constraint::Kind;

  /// Creates a constraint with an explicit kind (no kind inference).
  Constraint(const llvm::Record *record, Kind kind);
  /// Creates a constraint and infers the kind from the record's class
  /// hierarchy.
  Constraint(const llvm::Record *record);

  /// Records are pointer-comparable; override the ODS structural equality.
  bool operator==(const Constraint &that) const { return def == that.def; }
  bool operator!=(const Constraint &that) const { return def != that.def; }

  /// Returns the predicate for this constraint. This is TableGen-specific and
  /// requires access to the record.
  Pred getPredicate() const;

  /// Returns the underlying def.
  const llvm::Record &getDef() const { return *def; }

protected:
  const llvm::Record *def;

private:
  /// Tag type used to construct sentinel (empty/tombstone) Constraint objects
  /// for DenseMap without reading from a valid record.
  struct SentinelTag {};
  Constraint(SentinelTag, const llvm::Record *ptr, Kind kind);

  /// Populates all ODS base fields by reading from this->def and this->kind.
  /// Must be called only when def points to a valid llvm::Record.
  void populate();

  /// Returns the name of the base def for anonymous constraints, or
  /// std::nullopt if there is no base def.
  std::optional<std::string> getBaseDefName() const;

  friend struct llvm::DenseMapInfo<Constraint>;
};

// An constraint and the concrete entities to place the constraint on.
struct AppliedConstraint {
  AppliedConstraint(Constraint &&constraint, StringRef self,
                    std::vector<std::string> &&entities);

  Constraint constraint;
  // The symbol to replace `$_self` special placeholder in the constraint.
  std::string self;
  // The symbols to replace `$N` positional placeholders in the constraint.
  std::vector<std::string> entities;
};

} // namespace tblgen
} // namespace mlir

namespace llvm {
/// Unique constraints by their predicate and summary. Constraints that share
/// the same predicate may have different descriptions; ensure that the
/// correct error message is reported when verification fails.
template <>
struct DenseMapInfo<mlir::tblgen::Constraint> {
  using RecordDenseMapInfo = llvm::DenseMapInfo<const llvm::Record *>;

  static mlir::tblgen::Constraint getEmptyKey();
  static mlir::tblgen::Constraint getTombstoneKey();
  static unsigned getHashValue(mlir::tblgen::Constraint constraint);
  static bool isEqual(mlir::tblgen::Constraint lhs,
                      mlir::tblgen::Constraint rhs);
};
} // namespace llvm

#endif // MLIR_TABLEGEN_CONSTRAINT_H_
