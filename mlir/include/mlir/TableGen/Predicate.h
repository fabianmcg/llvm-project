//===- Predicate.h - Predicate class ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TableGen wrappers around ODS predicate classes. Each tblgen class derives
// from its mlir::ods counterpart, additionally stores a const llvm::Record*,
// and overrides getCondition() to read from the record rather than from
// pre-built ODS fields.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_PREDICATE_H_
#define MLIR_TABLEGEN_PREDICATE_H_

#include "mlir/ODS/Predicate.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/Hashing.h"

namespace llvm {
class Init;
class ListInit;
class Record;
class SMLoc;
} // namespace llvm

namespace mlir {
namespace tblgen {

// TableGen wrapper for a logical predicate. Derives from mlir::ods::Pred and
// additionally stores the underlying llvm::Record. Equality and hashing use
// pointer identity on the record (TableGen records are globally unique per
// compilation). getCondition() reads from the record directly and overrides
// the ods implementation so that this object need not pre-populate ODS fields.
class Pred : public mlir::ods::Pred {
public:
  // Constructs the null predicate.
  explicit Pred() {}
  // Construct a Pred from a TableGen record.
  explicit Pred(const llvm::Record *record);
  // Construct a Pred from a TableGen initializer.
  explicit Pred(const llvm::Init *init);

  // Check if the predicate is defined.
  bool isNull() const { return def == nullptr; }

  // Get the condition by dispatching through the record type, overriding the
  // ODS implementation so that we do not need pre-built child objects.
  std::string getCondition() const override;

  // Records are pointer-comparable.
  bool operator==(const Pred &other) const { return def == other.def; }
  bool operator!=(const Pred &other) const { return def != other.def; }

  // Return true if the predicate is not null.
  operator bool() const { return def; }

  // Hash a predicate by its pointer value.
  friend llvm::hash_code hash_value(Pred pred) {
    return llvm::hash_value(pred.def);
  }

  // Get the location of the predicate record.
  ArrayRef<SMLoc> getLoc() const;

  /// Return the underlying def.
  const llvm::Record &getDef() const { return *def; }

protected:
  // The TableGen definition of this predicate.
  const llvm::Record *def{nullptr};
};

// TableGen wrapper for a C-expression predicate. Derives from
// mlir::ods::CPred; the expression field is populated from the record in the
// constructor so the inherited getCondition() works correctly.
class CPred : public mlir::ods::CPred {
public:
  // Construct a CPred from a record.
  explicit CPred(const llvm::Record *record);
  // Construct a CPred from an initializer.
  explicit CPred(const llvm::Init *init);

  /// Return the underlying def.
  const llvm::Record &getDef() const { return *def; }

protected:
  const llvm::Record *def{nullptr};
};

// TableGen wrapper for a combined predicate. Derives from
// mlir::ods::CombinedPred. getCondition() is overridden to read child
// records lazily from the record instead of requiring pre-built ODS children.
class CombinedPred : public mlir::ods::CombinedPred {
public:
  // Construct a CombinedPred from a record.
  explicit CombinedPred(const llvm::Record *record);
  // Construct a CombinedPred from an initializer.
  explicit CombinedPred(const llvm::Init *init);

  // Override to use record-based tree building.
  std::string getCondition() const override;

  // Get the definition of the combiner kind record.
  const llvm::Record *getCombinerDef() const;

  // Get the child predicate records.
  std::vector<const llvm::Record *> getChildren() const;

  /// Return the underlying def.
  const llvm::Record &getDef() const { return *def; }

protected:
  const llvm::Record *def{nullptr};
};

// TableGen wrapper for a SubstLeaves combined predicate.
class SubstLeavesPred : public mlir::ods::SubstLeavesPred {
public:
  explicit SubstLeavesPred(const llvm::Record *record);

  // Override to use record-based tree building.
  std::string getCondition() const override;

  /// Return the underlying def.
  const llvm::Record &getDef() const { return *def; }

protected:
  const llvm::Record *def{nullptr};
};

// TableGen wrapper for a Concat combined predicate.
class ConcatPred : public mlir::ods::ConcatPred {
public:
  explicit ConcatPred(const llvm::Record *record);

  // Override to use record-based tree building.
  std::string getCondition() const override;

  /// Return the underlying def.
  const llvm::Record &getDef() const { return *def; }

protected:
  const llvm::Record *def{nullptr};
};

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_PREDICATE_H_
