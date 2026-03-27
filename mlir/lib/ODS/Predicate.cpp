//===- Predicate.cpp - ODS predicate classes ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/ODS/Predicate.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::ods;
using llvm::SpecificBumpPtrAllocator;

//===----------------------------------------------------------------------===//
// CombinedPred condition building
//===----------------------------------------------------------------------===//

namespace {

// Kinds of nodes in a logical predicate tree. Matches ods::Pred::Kind but
// adds True/False for simplification.
enum class PredCombinerKind {
  Leaf,
  And,
  Or,
  Not,
  SubstLeaves,
  Concat,
  // Special kinds used during simplification.
  False,
  True
};

// A node in a logical predicate tree.
struct PredNode {
  PredCombinerKind kind;
  const Pred *predicate;
  SmallVector<PredNode *, 4> children;
  std::string expr;

  // Prefix and suffix are used by ConcatPred.
  std::string prefix;
  std::string suffix;
};

} // namespace

// Map an ods::Pred::Kind to PredCombinerKind for tree building.
static PredCombinerKind getPredCombinerKind(const Pred &pred) {
  switch (pred.getKind()) {
  case Pred::Kind::Null:
  case Pred::Kind::CPred:
    return PredCombinerKind::Leaf;
  case Pred::Kind::And:
    return PredCombinerKind::And;
  case Pred::Kind::Or:
    return PredCombinerKind::Or;
  case Pred::Kind::Not:
    return PredCombinerKind::Not;
  case Pred::Kind::SubstLeaves:
    return PredCombinerKind::SubstLeaves;
  case Pred::Kind::Concat:
    return PredCombinerKind::Concat;
  }
  llvm_unreachable("unhandled Pred::Kind");
}

namespace {
// Substitution<pattern, replacement>.
using Subst = std::pair<StringRef, StringRef>;
} // namespace

// Perform the given substitutions on 'str' in-place.
static void performSubstitutions(std::string &str,
                                 ArrayRef<Subst> substitutions) {
  // Apply all parent substitutions from innermost to outermost.
  for (const auto &subst : llvm::reverse(substitutions)) {
    auto pos = str.find(subst.first);
    while (pos != std::string::npos) {
      str.replace(pos, subst.first.size(), std::string(subst.second));
      // Skip the newly inserted substring, which itself may contain the
      // pattern.
      pos += subst.second.size();
      pos = str.find(subst.first, pos);
    }
  }
}

// Build the predicate tree starting from the top-level predicate. All nodes
// are allocated within "allocator".
static PredNode *
buildPredicateTree(const Pred &root,
                   SpecificBumpPtrAllocator<PredNode> &allocator,
                   ArrayRef<Subst> substitutions) {
  auto *rootNode = allocator.Allocate();
  new (rootNode) PredNode;
  rootNode->kind = getPredCombinerKind(root);
  rootNode->predicate = &root;
  if (!root.isCombined()) {
    rootNode->expr = root.getCondition();
    performSubstitutions(rootNode->expr, substitutions);
    return rootNode;
  }

  // If the current combined predicate is a leaf substitution, append it to
  // the list before continuing.
  auto allSubstitutions = llvm::to_vector<4>(substitutions);
  if (rootNode->kind == PredCombinerKind::SubstLeaves) {
    const auto &substPred = static_cast<const SubstLeavesPred &>(root);
    allSubstitutions.push_back(
        {substPred.getPattern(), substPred.getReplacement()});
  } else if (rootNode->kind == PredCombinerKind::Concat) {
    const auto &concatPred = static_cast<const ConcatPred &>(root);
    rootNode->prefix = std::string(concatPred.getPrefix());
    performSubstitutions(rootNode->prefix, substitutions);
    rootNode->suffix = std::string(concatPred.getSuffix());
    performSubstitutions(rootNode->suffix, substitutions);
  }

  // Build child subtrees.
  const auto &combined = static_cast<const CombinedPred &>(root);
  for (const Pred *child : combined.getChildPreds()) {
    PredNode *childTree =
        buildPredicateTree(*child, allocator, allSubstitutions);
    rootNode->children.push_back(childTree);
  }
  return rootNode;
}

// Simplify a predicate tree by propagating known true/false predicates.
static PredNode *
propagateGroundTruth(PredNode *node,
                     const llvm::SmallPtrSetImpl<const Pred *> &knownTruePreds,
                     const llvm::SmallPtrSetImpl<const Pred *> &knownFalsePreds) {
  if (knownTruePreds.count(node->predicate) != 0) {
    node->kind = PredCombinerKind::True;
    node->children.clear();
    return node;
  }
  if (knownFalsePreds.count(node->predicate) != 0) {
    node->kind = PredCombinerKind::False;
    node->children.clear();
    return node;
  }

  // Stop recursion at substitution boundaries: the expressions in the leaves
  // below have been rewritten, so the original predicate ground truth no
  // longer applies.
  if (node->kind == PredCombinerKind::SubstLeaves)
    return node;

  if (node->kind == PredCombinerKind::And && node->children.empty()) {
    node->kind = PredCombinerKind::True;
    return node;
  }

  if (node->kind == PredCombinerKind::Or && node->children.empty()) {
    node->kind = PredCombinerKind::False;
    return node;
  }

  SmallVector<PredNode *, 4> children;
  std::swap(node->children, children);

  for (PredNode *child : children) {
    PredNode *simplifiedChild =
        propagateGroundTruth(child, knownTruePreds, knownFalsePreds);

    if (node->kind != PredCombinerKind::And &&
        node->kind != PredCombinerKind::Or) {
      node->children.push_back(simplifiedChild);
      continue;
    }

    auto collapseKind = node->kind == PredCombinerKind::And
                            ? PredCombinerKind::False
                            : PredCombinerKind::True;
    auto eraseKind = node->kind == PredCombinerKind::And
                         ? PredCombinerKind::True
                         : PredCombinerKind::False;
    const auto &collapseList =
        node->kind == PredCombinerKind::And ? knownFalsePreds : knownTruePreds;
    const auto &eraseList =
        node->kind == PredCombinerKind::And ? knownTruePreds : knownFalsePreds;
    if (simplifiedChild->kind == collapseKind ||
        collapseList.count(simplifiedChild->predicate) != 0) {
      node->kind = collapseKind;
      node->children.clear();
      return node;
    }
    if (simplifiedChild->kind == eraseKind ||
        eraseList.count(simplifiedChild->predicate) != 0)
      continue;
    node->children.push_back(simplifiedChild);
  }
  return node;
}

// Combine a list of predicate expressions using a binary combiner.
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

// Prepend negation to the single child predicate expression.
static std::string combineNot(ArrayRef<std::string> children) {
  assert(children.size() == 1 && "expected exactly one child predicate of Not");
  return (Twine("!(") + children.front() + Twine(')')).str();
}

// Recursively build the final condition expression from the predicate tree.
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

std::string CombinedPred::getCondition() const {
  SpecificBumpPtrAllocator<PredNode> allocator;
  PredNode *predicateTree = buildPredicateTree(*this, allocator, {});
  predicateTree =
      propagateGroundTruth(predicateTree,
                           /*knownTruePreds=*/llvm::SmallPtrSet<const Pred *, 2>(),
                           /*knownFalsePreds=*/llvm::SmallPtrSet<const Pred *, 2>());
  return getCombinedCondition(*predicateTree);
}
