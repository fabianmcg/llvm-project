//===- Predicate.h - ODS predicate classes ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines ODS predicate classes independent of LLVM TableGen.
// These classes model the predicates defined in mlir/include/mlir/IR/Constraints.td
// without any dependency on llvm::Record or related TableGen types.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ODS_PREDICATE_H_
#define MLIR_ODS_PREDICATE_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace mlir {
namespace ods {

// A logical predicate. Models the TableGen 'Pred' class hierarchy from
// mlir/include/mlir/IR/Constraints.td without depending on llvm::Record.
//
// getCondition() is virtual so that TableGen wrappers (mlir::tblgen::Pred)
// can override it to read from llvm::Record instead of explicit ODS fields.
class Pred {
public:
  // The kind of predicate, matching the TableGen 'PredCombinerKind' enum
  // plus CPred and Null.
  enum class Kind {
    Null,
    CPred,
    And,
    Or,
    Not,
    SubstLeaves,
    Concat,
  };

  // Constructs the null predicate (always true / undefined).
  Pred() : kind(Kind::Null) {}

  virtual ~Pred() = default;

  // Check if the predicate is defined.
  bool isNull() const { return kind == Kind::Null; }

  // Whether the predicate is a combination of other predicates.
  bool isCombined() const {
    return kind == Kind::And || kind == Kind::Or || kind == Kind::Not ||
           kind == Kind::SubstLeaves || kind == Kind::Concat;
  }

  // Get the predicate condition string. Virtual to allow TableGen wrappers to
  // override with record-based access.
  virtual std::string getCondition() const = 0;

  // Structural equality: compares kind and all stored fields.
  virtual bool operator==(const Pred &other) const { return kind == other.kind; }
  bool operator!=(const Pred &other) const { return !(*this == other); }

  // Return true if the predicate is not null.
  explicit operator bool() const { return kind != Kind::Null; }

  Kind getKind() const { return kind; }

protected:
  explicit Pred(Kind kind) : kind(kind) {}

  Kind kind;
};

// A predicate wrapping a C++ expression string. Models the TableGen 'CPred'
// class.
class CPred : public Pred {
public:
  explicit CPred(StringRef expression)
      : Pred(Kind::CPred), expr(expression) {}

  std::string getCondition() const override { return expr; }

  bool operator==(const Pred &other) const override {
    if (other.getKind() != Kind::CPred)
      return false;
    return expr == static_cast<const CPred &>(other).expr;
  }

  StringRef getExpression() const { return expr; }

private:
  std::string expr;
};

// A predicate that combines child predicates. Models the TableGen
// 'CombinedPred' class. Children are non-owning pointers; their lifetime
// must be managed by an ODSContext.
class CombinedPred : public Pred {
public:
  CombinedPred(Kind kind, ArrayRef<const Pred *> children)
      : Pred(kind), children(children.begin(), children.end()) {}

  // Get the predicate condition string by recursively combining children.
  std::string getCondition() const override;

  bool operator==(const Pred &other) const override {
    if (other.getKind() != kind)
      return false;
    return children == static_cast<const CombinedPred &>(other).children;
  }

  // Get the child predicates.
  ArrayRef<const Pred *> getChildPreds() const { return children; }

protected:
  SmallVector<const Pred *, 4> children;
};

// A combined predicate that rewrites the C expression in all 'CPred' leaves
// using a string substitution. Models the TableGen 'SubstLeavesPred' class.
class SubstLeavesPred : public CombinedPred {
public:
  SubstLeavesPred(StringRef pattern, StringRef replacement,
                  ArrayRef<const Pred *> children)
      : CombinedPred(Kind::SubstLeaves, children), pattern(pattern),
        replacement(replacement) {}

  StringRef getPattern() const { return pattern; }
  StringRef getReplacement() const { return replacement; }

private:
  std::string pattern;
  std::string replacement;
};

// A combined predicate that prepends a prefix and appends a suffix to the
// condition of its single child predicate. Models the TableGen 'ConcatPred'.
class ConcatPred : public CombinedPred {
public:
  ConcatPred(StringRef prefix, StringRef suffix,
             ArrayRef<const Pred *> children)
      : CombinedPred(Kind::Concat, children), prefix(prefix), suffix(suffix) {}

  StringRef getPrefix() const { return prefix; }
  StringRef getSuffix() const { return suffix; }

private:
  std::string prefix;
  std::string suffix;
};

} // namespace ods
} // namespace mlir

#endif // MLIR_ODS_PREDICATE_H_
