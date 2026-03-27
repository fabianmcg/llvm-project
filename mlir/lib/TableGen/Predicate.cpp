//===- Predicate.cpp - predFromRecord / predFromInit free functions -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements tblgen::predFromRecord() which eagerly computes the C++ condition
// string from a TableGen Pred record tree, then stores it in an ods::Pred.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Predicate.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Allocator.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using namespace mlir::tblgen;
using llvm::Init;
using llvm::Record;
using llvm::SpecificBumpPtrAllocator;

//===----------------------------------------------------------------------===//
// Record-based predicate tree building
//===----------------------------------------------------------------------===//

namespace {

enum class PredCombinerKind {
  Leaf,
  And,
  Or,
  Not,
  SubstLeaves,
  Concat,
  False,
  True
};

struct PredNode {
  PredCombinerKind kind;
  SmallVector<PredNode *, 4> children;
  std::string expr;
  std::string prefix;
  std::string suffix;
};

using Subst = std::pair<StringRef, StringRef>;

} // namespace

static PredCombinerKind getPredCombinerKind(const Record *record) {
  if (!record->isSubClassOf("CombinedPred"))
    return PredCombinerKind::Leaf;
  return llvm::StringSwitch<PredCombinerKind>(
             record->getValueAsDef("kind")->getName())
      .Case("PredCombinerAnd", PredCombinerKind::And)
      .Case("PredCombinerOr", PredCombinerKind::Or)
      .Case("PredCombinerNot", PredCombinerKind::Not)
      .Case("PredCombinerSubstLeaves", PredCombinerKind::SubstLeaves)
      .Case("PredCombinerConcat", PredCombinerKind::Concat)
      .Default(PredCombinerKind::Leaf);
}

static void performSubstitutions(std::string &str,
                                 ArrayRef<Subst> substitutions) {
  for (const auto &subst : llvm::reverse(substitutions)) {
    auto pos = str.find(subst.first);
    while (pos != std::string::npos) {
      str.replace(pos, subst.first.size(), std::string(subst.second));
      pos += subst.second.size();
      pos = str.find(subst.first, pos);
    }
  }
}

static PredNode *
buildPredicateTree(const Record *root,
                   SpecificBumpPtrAllocator<PredNode> &allocator,
                   ArrayRef<Subst> substitutions) {
  auto *rootNode = allocator.Allocate();
  new (rootNode) PredNode;
  rootNode->kind = getPredCombinerKind(root);

  if (!root->isSubClassOf("CombinedPred")) {
    rootNode->expr = std::string(root->getValueAsString("predExpr"));
    performSubstitutions(rootNode->expr, substitutions);
    return rootNode;
  }

  auto allSubstitutions = llvm::to_vector<4>(substitutions);
  if (rootNode->kind == PredCombinerKind::SubstLeaves) {
    allSubstitutions.push_back(
        {root->getValueAsString("pattern"),
         root->getValueAsString("replacement")});
  } else if (rootNode->kind == PredCombinerKind::Concat) {
    rootNode->prefix = std::string(root->getValueAsString("prefix"));
    performSubstitutions(rootNode->prefix, substitutions);
    rootNode->suffix = std::string(root->getValueAsString("suffix"));
    performSubstitutions(rootNode->suffix, substitutions);
  }

  for (const Record *child : root->getValueAsListOfDefs("children")) {
    PredNode *childTree = buildPredicateTree(child, allocator, allSubstitutions);
    rootNode->children.push_back(childTree);
  }
  return rootNode;
}

static std::string combineBinary(ArrayRef<std::string> children,
                                 const std::string &combiner,
                                 std::string init) {
  if (children.empty())
    return init;
  int size = static_cast<int>(children.size());
  if (size == 1)
    return children.front();
  std::string str;
  llvm::raw_string_ostream os(str);
  os << '(' << children.front() << ')';
  for (int i = 1; i < size; ++i)
    os << ' ' << combiner << " (" << children[i] << ')';
  return str;
}

static std::string combineNot(ArrayRef<std::string> children) {
  assert(children.size() == 1 && "expected exactly one child predicate of Not");
  return (Twine("!(") + children.front() + Twine(')')).str();
}

static std::string getCombinedCondition(const PredNode &root) {
  if (root.kind == PredCombinerKind::Leaf)
    return root.expr;
  if (root.kind == PredCombinerKind::True)
    return "true";
  if (root.kind == PredCombinerKind::False)
    return "false";

  SmallVector<std::string, 4> childExpressions;
  childExpressions.reserve(root.children.size());
  for (const PredNode *child : root.children)
    childExpressions.push_back(getCombinedCondition(*child));

  if (root.kind == PredCombinerKind::And)
    return combineBinary(childExpressions, "&&", "true");
  if (root.kind == PredCombinerKind::Or)
    return combineBinary(childExpressions, "||", "false");
  if (root.kind == PredCombinerKind::Not)
    return combineNot(childExpressions);
  if (root.kind == PredCombinerKind::Concat) {
    assert(childExpressions.size() == 1 &&
           "ConcatPred should only have one child");
    return root.prefix + childExpressions.front() + root.suffix;
  }
  if (root.kind == PredCombinerKind::SubstLeaves) {
    assert(childExpressions.size() == 1 &&
           "substitution predicate must have one child");
    return childExpressions[0];
  }
  llvm_unreachable("unsupported predicate kind");
}

/// Compute the condition string from a TableGen Pred record.
static std::string computeCondition(const Record *record) {
  if (record->isSubClassOf("CPred"))
    return std::string(record->getValueAsString("predExpr"));
  if (record->isSubClassOf("CombinedPred")) {
    SpecificBumpPtrAllocator<PredNode> allocator;
    PredNode *tree = buildPredicateTree(record, allocator, {});
    return getCombinedCondition(*tree);
  }
  llvm_unreachable("unsupported Pred subclass");
}

//===----------------------------------------------------------------------===//
// Free functions
//===----------------------------------------------------------------------===//

ods::Pred tblgen::predFromRecord(const llvm::Record *record) {
  if (!record)
    return ods::Pred();
  assert(record->isSubClassOf("Pred") &&
         "predFromRecord requires a subclass of TableGen 'Pred'");
  return ods::Pred(computeCondition(record));
}

ods::Pred tblgen::predFromInit(const llvm::Init *init) {
  if (!init)
    return ods::Pred();
  const auto *defInit = dyn_cast<llvm::DefInit>(init);
  if (!defInit)
    return ods::Pred();
  return predFromRecord(defInit->getDef());
}
