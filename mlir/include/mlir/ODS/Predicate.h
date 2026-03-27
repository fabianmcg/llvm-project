//===- Predicate.h - ODS predicate value type ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ods::Pred value type which models a logical predicate
// independently of LLVM TableGen. Pred stores an eagerly-computed C++
// condition string; condition building from TableGen records happens in the
// tblgen::predFromRecord() free function.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ODS_PREDICATE_H_
#define MLIR_ODS_PREDICATE_H_

#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace mlir {
namespace ods {

/// A logical predicate. Stores a precomputed C++ condition string with no
/// dependency on LLVM TableGen types. The default-constructed Pred is null
/// (represents the always-true / undefined predicate).
class Pred {
public:
  /// Constructs a null (undefined) predicate.
  Pred() = default;

  /// Constructs a non-null predicate from a precomputed condition string.
  explicit Pred(llvm::StringRef condition)
      : null(false), conditionStr(condition.str()) {}

  /// Returns true if this predicate is undefined (null / always-true).
  bool isNull() const { return null; }

  /// Returns the C++ condition string. Must not be called on a null predicate.
  std::string getCondition() const { return conditionStr; }

  /// Returns true if the predicate is not null.
  explicit operator bool() const { return !null; }

  bool operator==(const Pred &other) const {
    return null == other.null && conditionStr == other.conditionStr;
  }
  bool operator!=(const Pred &other) const { return !(*this == other); }

  friend llvm::hash_code hash_value(const Pred &pred) {
    return llvm::hash_combine(pred.null, pred.conditionStr);
  }

// All fields are public so that free factory functions (e.g.,
// tblgen::predFromRecord) can populate them without requiring friendship.
// Callers should use the accessor methods above for read access.
public:
  bool null{true};
  std::string conditionStr;
};

} // namespace ods
} // namespace mlir

#endif // MLIR_ODS_PREDICATE_H_
