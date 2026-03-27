//===- Predicate.cpp - Predicate class ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TableGen wrappers around ODS predicate classes.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Predicate.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using namespace tblgen;
using llvm::Init;
using llvm::Record;
using llvm::SpecificBumpPtrAllocator;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Derive the ods::Pred::Kind from a TableGen record.
static ods::Pred::Kind getKindFromRecord(const Record *record) {
  if (!record)
    return ods::Pred::Kind::Null;
  if (record->isSubClassOf("CPred"))
    return ods::Pred::Kind::CPred;
  if (record->isSubClassOf("CombinedPred")) {
    return llvm::StringSwitch<ods::Pred::Kind>(
               record->getValueAsDef("kind")->getName())
        .Case("PredCombinerAnd", ods::Pred::Kind::And)
        .Case("PredCombinerOr", ods::Pred::Kind::Or)
        .Case("PredCombinerNot", ods::Pred::Kind::Not)
        .Case("PredCombinerSubstLeaves", ods::Pred::Kind::SubstLeaves)
        .Case("PredCombinerConcat", ods::Pred::Kind::Concat)
        .Default(ods::Pred::Kind::Null);
  }
  return ods::Pred::Kind::Null;
}

// Extract the llvm::Record from an Init, or return nullptr.
static const Record *recordFromInit(const Init *init) {
  if (const auto *defInit = dyn_cast_or_null<llvm::DefInit>(init))
    return defInit->getDef();
  return nullptr;
}

// Extract the CPred expression string from an Init, or return "".
static std::string cpredExprFromInit(const Init *init) {
  const Record *rec = recordFromInit(init);
  if (rec && rec->isSubClassOf("CPred"))
    return std::string(rec->getValueAsString("predExpr"));
  return "";
}

//===----------------------------------------------------------------------===//
// Pred
//===----------------------------------------------------------------------===//

Pred::Pred(const Record *record) : def(record) {
  assert(def->isSubClassOf("Pred") &&
         "must be a subclass of TableGen 'Pred' class");
  kind = getKindFromRecord(def);
}

Pred::Pred(const Init *init) {
  if (const auto *defInit = dyn_cast_or_null<llvm::DefInit>(init)) {
    def = defInit->getDef();
    kind = getKindFromRecord(def);
  }
}

ArrayRef<SMLoc> Pred::getLoc() const { return def->getLoc(); }

std::string Pred::getCondition() const {
  assert(!isNull() && "null predicate does not have a condition");
  // Dispatch to the appropriate subclass based on the record type.
  if (def->isSubClassOf("CombinedPred"))
    return CombinedPred(def).getCondition();
  if (def->isSubClassOf("CPred"))
    return std::string(def->getValueAsString("predExpr"));
  llvm_unreachable("unsupported predicate type in Pred::getCondition");
}

//===----------------------------------------------------------------------===//
// CPred
//===----------------------------------------------------------------------===//

CPred::CPred(const Record *record)
    : ods::CPred(record->getValueAsString("predExpr")), def(record) {
  assert(def->isSubClassOf("CPred") &&
         "must be a subclass of TableGen 'CPred' class");
}

CPred::CPred(const Init *init) : ods::CPred(cpredExprFromInit(init)) {
  def = recordFromInit(init);
  assert((!def || def->isSubClassOf("CPred")) &&
         "must be a subclass of TableGen 'CPred' class");
}

//===----------------------------------------------------------------------===//
// Record-based tree building for CombinedPred::getCondition()
//
// This keeps the original buildPredicateTree logic working from records,
// so that CombinedPred does not need pre-built ODS children.
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
    // Leaf: CPred or similar.
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

//===----------------------------------------------------------------------===//
// CombinedPred
//===----------------------------------------------------------------------===//

CombinedPred::CombinedPred(const Record *record)
    : ods::CombinedPred(getKindFromRecord(record), {}), def(record) {
  assert(def->isSubClassOf("CombinedPred") &&
         "must be a subclass of TableGen 'CombinedPred' class");
}

CombinedPred::CombinedPred(const Init *init)
    : ods::CombinedPred(ods::Pred::Kind::Null, {}) {
  if (const auto *defInit = dyn_cast_or_null<llvm::DefInit>(init)) {
    def = defInit->getDef();
    assert((!def || def->isSubClassOf("CombinedPred")) &&
           "must be a subclass of TableGen 'CombinedPred' class");
    if (def)
      kind = getKindFromRecord(def);
  }
}

std::string CombinedPred::getCondition() const {
  SpecificBumpPtrAllocator<PredNode> allocator;
  PredNode *predicateTree = buildPredicateTree(def, allocator, {});
  return getCombinedCondition(*predicateTree);
}

const Record *CombinedPred::getCombinerDef() const {
  assert(def->getValue("kind") && "CombinedPred must have a value 'kind'");
  return def->getValueAsDef("kind");
}

std::vector<const Record *> CombinedPred::getChildren() const {
  assert(def->getValue("children") &&
         "CombinedPred must have a value 'children'");
  return def->getValueAsListOfDefs("children");
}

//===----------------------------------------------------------------------===//
// SubstLeavesPred
//===----------------------------------------------------------------------===//

SubstLeavesPred::SubstLeavesPred(const Record *record)
    : ods::SubstLeavesPred(record->getValueAsString("pattern"),
                           record->getValueAsString("replacement"), {}),
      def(record) {}

std::string SubstLeavesPred::getCondition() const {
  SpecificBumpPtrAllocator<PredNode> allocator;
  PredNode *predicateTree = buildPredicateTree(def, allocator, {});
  return getCombinedCondition(*predicateTree);
}

//===----------------------------------------------------------------------===//
// ConcatPred
//===----------------------------------------------------------------------===//

ConcatPred::ConcatPred(const Record *record)
    : ods::ConcatPred(record->getValueAsString("prefix"),
                      record->getValueAsString("suffix"), {}),
      def(record) {}

std::string ConcatPred::getCondition() const {
  SpecificBumpPtrAllocator<PredNode> allocator;
  PredNode *predicateTree = buildPredicateTree(def, allocator, {});
  return getCombinedCondition(*predicateTree);
}
