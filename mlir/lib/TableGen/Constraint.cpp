//===- Constraint.cpp - Constraint class ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Constraint.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using namespace mlir::tblgen;

//===----------------------------------------------------------------------===//
// Constraint
//===----------------------------------------------------------------------===//

// Private constructor for DenseMap sentinels: stores the pointer without
// reading from the record.
Constraint::Constraint(SentinelTag, const llvm::Record *ptr, Kind kind)
    : mlir::ods::Constraint(kind), def(ptr) {}

Constraint::Constraint(const llvm::Record *record, Kind kind)
    : Constraint(SentinelTag{}, record, kind) {
  if (def)
    populate();
}

Constraint::Constraint(const llvm::Record *record)
    : Constraint(SentinelTag{}, record, CK_Uncategorized) {
  // Look through OpVariable's to their constraint.
  if (def->isSubClassOf("OpVariable"))
    def = def->getValueAsDef("constraint");

  if (def->isSubClassOf("TypeConstraint")) {
    kind = CK_Type;
  } else if (def->isSubClassOf("AttrConstraint")) {
    kind = CK_Attr;
  } else if (def->isSubClassOf("PropConstraint")) {
    kind = CK_Prop;
  } else if (def->isSubClassOf("RegionConstraint")) {
    kind = CK_Region;
  } else if (def->isSubClassOf("SuccessorConstraint")) {
    kind = CK_Successor;
  } else if (!def->isSubClassOf("Constraint")) {
    llvm::errs() << "Expected a constraint but got: \n" << *def << "\n";
    llvm::report_fatal_error("Abort");
  }

  populate();
}

void Constraint::populate() {
  // summary
  if (std::optional<StringRef> s = def->getValueAsOptionalString("summary"))
    summary = s->str();
  else
    summary = def->getName().str();

  // description
  description =
      def->getValueAsOptionalString("description").value_or("").str();

  // conditionTemplate: computed from the predicate, same logic as getPredicate()
  auto *predVal = def->getValue("predicate");
  if (predVal) {
    if (const auto *pred = dyn_cast<llvm::DefInit>(predVal->getValue()))
      conditionTemplate = Pred(pred).getCondition();
  }

  // defName: may use the base def's name for anonymous constraints
  std::optional<std::string> baseDefName = getBaseDefName();
  if (baseDefName)
    defName = *baseDefName;
  else
    defName = def->getName().str();

  // uniqueDefName
  std::string name = def->getName().str();
  if (!def->isAnonymous()) {
    uniqueDefName = name;
  } else {
    if (baseDefName)
      uniqueDefName = (*baseDefName + "(" + name + ")");
    else
      uniqueDefName = name;
  }

  // cppFunctionName
  std::optional<StringRef> cppName =
      def->getValueAsOptionalString("cppFunctionName");
  if (cppName && !cppName->empty())
    cppFunctionName = cppName->str();

  // variadic: kind-specific subclass check
  switch (kind) {
  case CK_Region:
    variadic = def->isSubClassOf("VariadicRegion");
    break;
  case CK_Successor:
    variadic = def->isSubClassOf("VariadicSuccessor");
    break;
  case CK_Type:
    variadic = def->isSubClassOf("Variadic");
    break;
  default:
    variadic = false;
    break;
  }
}

Pred Constraint::getPredicate() const {
  auto *val = def->getValue("predicate");

  // If no predicate is specified, then return the null predicate (which
  // corresponds to true).
  if (!val)
    return Pred();

  const auto *pred = dyn_cast<llvm::DefInit>(val->getValue());
  return Pred(pred);
}

std::optional<std::string> Constraint::getBaseDefName() const {
  // Functor used to check a base def in the case where the current def is
  // anonymous. Returns the base def name as an owned string so that it
  // remains valid after the temporary Constraint is destroyed.
  auto checkBaseDefFn =
      [&](StringRef baseName) -> std::optional<std::string> {
    if (const auto *defValue = def->getValue(baseName)) {
      if (const auto *defInit = dyn_cast<llvm::DefInit>(defValue->getValue()))
        return Constraint(defInit->getDef(), kind).getDefName().str();
    }
    return std::nullopt;
  };

  switch (kind) {
  case CK_Attr:
    if (def->isAnonymous())
      return checkBaseDefFn("baseAttr");
    return std::nullopt;
  case CK_Type:
    if (def->isAnonymous())
      return checkBaseDefFn("baseType");
    return std::nullopt;
  default:
    return std::nullopt;
  }
}

AppliedConstraint::AppliedConstraint(Constraint &&constraint,
                                     llvm::StringRef self,
                                     std::vector<std::string> &&entities)
    : constraint(constraint), self(std::string(self)),
      entities(std::move(entities)) {}

Constraint DenseMapInfo<Constraint>::getEmptyKey() {
  return Constraint(Constraint::SentinelTag{},
                    RecordDenseMapInfo::getEmptyKey(),
                    Constraint::CK_Uncategorized);
}

Constraint DenseMapInfo<Constraint>::getTombstoneKey() {
  return Constraint(Constraint::SentinelTag{},
                    RecordDenseMapInfo::getTombstoneKey(),
                    Constraint::CK_Uncategorized);
}

unsigned DenseMapInfo<Constraint>::getHashValue(Constraint constraint) {
  if (constraint == getEmptyKey())
    return RecordDenseMapInfo::getHashValue(RecordDenseMapInfo::getEmptyKey());
  if (constraint == getTombstoneKey()) {
    return RecordDenseMapInfo::getHashValue(
        RecordDenseMapInfo::getTombstoneKey());
  }
  return llvm::hash_combine(constraint.getPredicate(), constraint.getSummary());
}

bool DenseMapInfo<Constraint>::isEqual(Constraint lhs, Constraint rhs) {
  if (lhs == rhs)
    return true;
  if (lhs == getEmptyKey() || lhs == getTombstoneKey())
    return false;
  if (rhs == getEmptyKey() || rhs == getTombstoneKey())
    return false;
  return lhs.getPredicate() == rhs.getPredicate() &&
         lhs.getSummary() == rhs.getSummary();
}
