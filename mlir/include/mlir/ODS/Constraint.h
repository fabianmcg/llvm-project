//===- Constraint.h - ODS Constraint class ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ODS Constraint class, which models a MLIR constraint
// definition independently of LLVM TableGen.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ODS_CONSTRAINT_H_
#define MLIR_ODS_CONSTRAINT_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringRef.h"

#include <optional>
#include <string>

namespace mlir {
namespace ods {

/// Models a constraint definition, storing all data as plain C++ fields with
/// no dependency on LLVM TableGen types.
class Constraint {
public:
  /// The kind of constraint this represents.
  enum Kind {
    CK_Attr,
    CK_Prop,
    CK_Region,
    CK_Successor,
    CK_Type,
    CK_Uncategorized,
    /// Sentinel values used by DenseMapInfo. Not valid for real constraints.
    CK_DenseMapEmpty,
    CK_DenseMapTombstone,
  };

  /// Constructs a constraint of the given kind with all other fields
  /// default-initialised. Callers (including free factory functions) may then
  /// populate the public fields directly.
  explicit Constraint(Kind kind) : kind(kind) {}

  Kind getKind() const { return kind; }

  /// Returns the user-readable summary of this constraint. If no summary was
  /// provided, returns the def name.
  StringRef getSummary() const { return summary; }

  /// Returns the long-form description of this constraint.
  StringRef getDescription() const { return description; }

  /// Returns the condition template that can be used to check if a type or
  /// attribute satisfies this constraint. May contain "{0}" that must be
  /// substituted with an expression returning an mlir::Type or mlir::Attribute.
  std::string getConditionTemplate() const { return conditionTemplate; }

  /// Returns the name of the TableGen def of this constraint. For anonymous
  /// defs, returns the name of the base def if one is present.
  StringRef getDefName() const { return defName; }

  /// Returns a unique name for the TableGen def of this constraint. For
  /// anonymous defs, attaches the name of the base def when present.
  std::string getUniqueDefName() const { return uniqueDefName; }

  /// Returns the name of the C++ function that should be generated for this
  /// constraint, or std::nullopt if no C++ function should be generated.
  std::optional<StringRef> getCppFunctionName() const {
    if (!cppFunctionName)
      return std::nullopt;
    return StringRef(*cppFunctionName);
  }

  /// Returns true if this constraint is variadic. Applies to Region, Successor,
  /// and TypeConstraint subclasses; false for all others.
  bool isVariadic() const { return variadic; }

  bool operator==(const Constraint &other) const {
    return kind == other.kind && summary == other.summary &&
           conditionTemplate == other.conditionTemplate;
  }
  bool operator!=(const Constraint &other) const { return !(*this == other); }

  /// Returns the interface type for this constraint. Only meaningful for
  /// CK_Prop constraints; returns an empty string for all other kinds.
  StringRef getInterfaceType() const { return propInterfaceType; }

// All fields are public so that free factory functions can populate them
// without requiring friendship.  Callers should use the accessor methods above
// for read access.
public:
  Kind kind{CK_Uncategorized};
  std::string summary;
  std::string description;
  std::string conditionTemplate;
  std::string defName;
  std::string uniqueDefName;
  std::optional<std::string> cppFunctionName;
  bool variadic{false};
  /// Interface type string; only populated for CK_Prop constraints.
  std::string propInterfaceType;
};

} // namespace ods
} // namespace mlir

namespace llvm {
/// Allows mlir::ods::Constraint to be used as a DenseMap/DenseSet key.
/// Constraints are uniqued by (kind, conditionTemplate, summary).
/// CK_DenseMapEmpty and CK_DenseMapTombstone are reserved sentinel values.
template <>
struct DenseMapInfo<mlir::ods::Constraint> {
  using Kind = mlir::ods::Constraint::Kind;

  static mlir::ods::Constraint getEmptyKey() {
    return mlir::ods::Constraint(Kind::CK_DenseMapEmpty);
  }
  static mlir::ods::Constraint getTombstoneKey() {
    return mlir::ods::Constraint(Kind::CK_DenseMapTombstone);
  }
  static unsigned getHashValue(const mlir::ods::Constraint &c) {
    return llvm::hash_combine(c.getConditionTemplate(), c.getSummary());
  }
  static bool isEqual(const mlir::ods::Constraint &lhs,
                      const mlir::ods::Constraint &rhs) {
    if (lhs.getKind() == Kind::CK_DenseMapEmpty ||
        lhs.getKind() == Kind::CK_DenseMapTombstone)
      return lhs.getKind() == rhs.getKind();
    if (rhs.getKind() == Kind::CK_DenseMapEmpty ||
        rhs.getKind() == Kind::CK_DenseMapTombstone)
      return false;
    return lhs.getConditionTemplate() == rhs.getConditionTemplate() &&
           lhs.getSummary() == rhs.getSummary();
  }
};
} // namespace llvm

#endif // MLIR_ODS_CONSTRAINT_H_
