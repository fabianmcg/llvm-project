//===- AttrOrTypeDef.cpp - AttrOrTypeDef wrapper classes ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/AttrOrTypeDef.h"
#include "mlir/TableGen/Dialect.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using namespace mlir::tblgen;
using llvm::DefInit;
using llvm::Init;
using llvm::ListInit;
using llvm::Record;
using llvm::RecordVal;
using llvm::StringInit;

//===----------------------------------------------------------------------===//
// AttrOrTypeBuilder
//===----------------------------------------------------------------------===//

std::optional<StringRef> AttrOrTypeBuilder::getReturnType() const {
  std::optional<StringRef> type = def->getValueAsOptionalString("returnType");
  return type && !type->empty() ? type : std::nullopt;
}

bool AttrOrTypeBuilder::hasInferredContextParameter() const {
  return def->getValueAsBit("hasInferredContextParam");
}

//===----------------------------------------------------------------------===//
// AttrOrTypeDef
//===----------------------------------------------------------------------===//

AttrOrTypeDef::AttrOrTypeDef(const Record *def) : def(def) {
  // Populate the builders.
  const auto *builderList =
      dyn_cast_or_null<ListInit>(def->getValueInit("builders"));
  if (builderList && !builderList->empty()) {
    for (const Init *init : builderList->getElements()) {
      AttrOrTypeBuilder builder(cast<DefInit>(init)->getDef(), def->getLoc());

      // Ensure that all parameters have names.
      for (const AttrOrTypeBuilder::Parameter &param :
           builder.getParameters()) {
        if (!param.getName())
          PrintFatalError(def->getLoc(), "builder parameters must have a name");
      }
      builders.emplace_back(builder);
    }
  }

  // Populate the traits.
  if (auto *traitList = def->getValueAsListInit("traits")) {
    SmallPtrSet<const Init *, 32> traitSet;
    traits.reserve(traitSet.size());
    llvm::unique_function<void(const ListInit *)> processTraitList =
        [&](const ListInit *traitList) {
          for (auto *traitInit : *traitList) {
            if (!traitSet.insert(traitInit).second)
              continue;

            // If this is an interface, add any bases to the trait list.
            auto *traitDef = cast<DefInit>(traitInit)->getDef();
            if (traitDef->isSubClassOf("Interface")) {
              if (auto *bases = traitDef->getValueAsListInit("baseInterfaces"))
                processTraitList(bases);
            }

            traits.push_back(Trait::create(traitInit));
          }
        };
    processTraitList(traitList);
  }

  // Populate the parameters.
  if (auto *parametersDag = def->getValueAsDag("parameters")) {
    for (unsigned i = 0, e = parametersDag->getNumArgs(); i < e; ++i)
      parameters.push_back(AttrOrTypeParameter(parametersDag, i));
  }

  // Verify the use of the mnemonic field.
  bool hasCppFormat = def->getValueAsBit("hasCustomAssemblyFormat");
  bool hasDeclarativeFormat =
      def->getValueAsOptionalString("assemblyFormat").has_value();
  std::optional<StringRef> mnemonicVal =
      def->getValueAsOptionalString("mnemonic");
  if (mnemonicVal) {
    if (hasCppFormat && hasDeclarativeFormat) {
      PrintFatalError(def->getLoc(), "cannot specify both 'assemblyFormat' "
                                     "and 'hasCustomAssemblyFormat'");
    }
    if (!parameters.empty() && !hasCppFormat && !hasDeclarativeFormat) {
      PrintFatalError(def->getLoc(),
                      "must specify either 'assemblyFormat' or "
                      "'hasCustomAssemblyFormat' when 'mnemonic' is set");
    }
  } else if (hasCppFormat || hasDeclarativeFormat) {
    PrintFatalError(def->getLoc(),
                    "'assemblyFormat' or 'hasCustomAssemblyFormat' can only be "
                    "used when 'mnemonic' is set");
  }
  // Assembly format printer requires accessors to be generated.
  bool genAccessorsVal = def->getValueAsBit("genAccessors");
  if (hasDeclarativeFormat && !genAccessorsVal) {
    PrintFatalError(def->getLoc(),
                    "'assemblyFormat' requires 'genAccessors' to be true");
  }

  // Populate ods::AttrOrTypeDef scalar fields eagerly.
  {
    const auto *dialectInit =
        dyn_cast<DefInit>(def->getValue("dialect")->getValue());
    if (dialectInit)
      dialect = tblgen::dialectFromRecord(dialectInit->getDef());

    name = def->getName().str();
    cppClassName = def->getValueAsString("cppClassName").str();
    cppBaseClassName = def->getValueAsString("cppBaseClassName").str();

    const RecordVal *descVal = def->getValue("description");
    hasDescriptionFlag = descVal && isa<StringInit>(descVal->getValue());
    if (hasDescriptionFlag)
      description = def->getValueAsString("description").str();

    const RecordVal *summaryVal = def->getValue("summary");
    hasSummaryFlag = summaryVal && isa<StringInit>(summaryVal->getValue());
    if (hasSummaryFlag)
      summary = def->getValueAsString("summary").str();

    storageClassName = def->getValueAsString("storageClass").str();
    storageNamespace = def->getValueAsString("storageNamespace").str();
    genStorageClassFlag = def->getValueAsBit("genStorageClass");
    hasStorageCustomConstructorFlag =
        def->getValueAsBit("hasStorageCustomConstructor");

    if (mnemonicVal && !mnemonicVal->empty())
      mnemonic = mnemonicVal->str();

    hasCustomAssemblyFormatFlag = hasCppFormat;

    if (hasDeclarativeFormat)
      assemblyFormat =
          def->getValueAsOptionalString("assemblyFormat")->str();

    genAccessorsFlag = genAccessorsVal;
    genVerifyDeclFlag = def->getValueAsBit("genVerifyDecl");

    // Precompute genVerifyInvariantsImpl: true if any parameter has a
    // constraint or any trait is a PredTrait.
    verifyInvariantsImplFlag =
        any_of(parameters,
               [](const AttrOrTypeParameter &p) {
                 return p.getConstraint() != std::nullopt;
               }) ||
        any_of(traits, [](const Trait &t) { return isa<PredTrait>(&t); });

    StringRef extraDeclsStr = def->getValueAsString("extraClassDeclaration");
    if (!extraDeclsStr.empty())
      extraDecls = extraDeclsStr.str();

    StringRef extraDefsStr = def->getValueAsString("extraClassDefinition");
    if (!extraDefsStr.empty())
      extraDefs = extraDefsStr.str();

    genMnemonicAliasFlag = def->getValueAsBit("genMnemonicAlias");
    skipDefaultBuildersFlag = def->getValueAsBit("skipDefaultBuilders");
  }
  // TODO: Ensure that a suitable builder prototype can be generated:
  // https://llvm.org/PR56415
}

ods::Dialect AttrOrTypeDef::getDialect() const {
  const auto *dialectInit =
      dyn_cast<DefInit>(def->getValue("dialect")->getValue());
  return tblgen::dialectFromRecord(dialectInit ? dialectInit->getDef() : nullptr);
}

unsigned AttrOrTypeDef::getNumParameters() const {
  auto *parametersDag = def->getValueAsDag("parameters");
  return parametersDag ? parametersDag->getNumArgs() : 0;
}

ArrayRef<SMLoc> AttrOrTypeDef::getLoc() const { return def->getLoc(); }

bool AttrOrTypeDef::operator==(const AttrOrTypeDef &other) const {
  return def == other.def;
}

bool AttrOrTypeDef::operator<(const AttrOrTypeDef &other) const {
  return getName() < other.getName();
}

//===----------------------------------------------------------------------===//
// AttrDef
//===----------------------------------------------------------------------===//

std::optional<StringRef> AttrDef::getTypeBuilder() const {
  return def->getValueAsOptionalString("typeBuilder");
}

bool AttrDef::classof(const AttrOrTypeDef *def) {
  return def->getDef()->isSubClassOf("AttrDef");
}

StringRef AttrDef::getAttrName() const {
  return def->getValueAsString("attrName");
}

//===----------------------------------------------------------------------===//
// TypeDef
//===----------------------------------------------------------------------===//

bool TypeDef::classof(const AttrOrTypeDef *def) {
  return def->getDef()->isSubClassOf("TypeDef");
}

StringRef TypeDef::getTypeName() const {
  return def->getValueAsString("typeName");
}

//===----------------------------------------------------------------------===//
// AttrOrTypeParameter
//===----------------------------------------------------------------------===//

template <typename InitT>
auto AttrOrTypeParameter::getDefValue(StringRef name) const {
  std::optional<decltype(std::declval<InitT>().getValue())> result;
  if (const auto *param = dyn_cast<DefInit>(getDef()))
    if (const auto *init = param->getDef()->getValue(name))
      if (const auto *value = dyn_cast_or_null<InitT>(init->getValue()))
        result = value->getValue();
  return result;
}

bool AttrOrTypeParameter::isAnonymous() const {
  return !def->getArgName(index);
}

StringRef AttrOrTypeParameter::getName() const {
  return def->getArgName(index)->getValue();
}

std::string AttrOrTypeParameter::getAccessorName() const {
  return "get" +
         llvm::convertToCamelFromSnakeCase(getName(), /*capitalizeFirst=*/true);
}

std::optional<StringRef> AttrOrTypeParameter::getAllocator() const {
  return getDefValue<StringInit>("allocator");
}

bool AttrOrTypeParameter::hasCustomComparator() const {
  return getDefValue<StringInit>("comparator").has_value();
}

StringRef AttrOrTypeParameter::getComparator() const {
  return getDefValue<StringInit>("comparator").value_or("$_lhs == $_rhs");
}

StringRef AttrOrTypeParameter::getCppType() const {
  if (auto *stringType = dyn_cast<StringInit>(getDef()))
    return stringType->getValue();
  auto cppType = getDefValue<StringInit>("cppType");
  if (cppType)
    return *cppType;
  if (const auto *init = dyn_cast<DefInit>(getDef()))
    llvm::PrintFatalError(
        init->getDef()->getLoc(),
        Twine("Missing `cppType` field in Attribute/Type parameter: ") +
            init->getAsString());
  llvm::reportFatalUsageError(
      Twine("Missing `cppType` field in Attribute/Type parameter: ") +
      getDef()->getAsString());
}

StringRef AttrOrTypeParameter::getCppAccessorType() const {
  return getDefValue<StringInit>("cppAccessorType").value_or(getCppType());
}

StringRef AttrOrTypeParameter::getCppStorageType() const {
  return getDefValue<StringInit>("cppStorageType").value_or(getCppType());
}

StringRef AttrOrTypeParameter::getConvertFromStorage() const {
  return getDefValue<StringInit>("convertFromStorage").value_or("$_self");
}

std::optional<StringRef> AttrOrTypeParameter::getParser() const {
  return getDefValue<StringInit>("parser");
}

std::optional<StringRef> AttrOrTypeParameter::getPrinter() const {
  return getDefValue<StringInit>("printer");
}

std::optional<StringRef> AttrOrTypeParameter::getSummary() const {
  return getDefValue<StringInit>("summary");
}

StringRef AttrOrTypeParameter::getSyntax() const {
  if (auto *stringType = dyn_cast<StringInit>(getDef()))
    return stringType->getValue();
  return getDefValue<StringInit>("syntax").value_or(getCppType());
}

bool AttrOrTypeParameter::isOptional() const {
  return getDefaultValue().has_value();
}

std::optional<StringRef> AttrOrTypeParameter::getDefaultValue() const {
  std::optional<StringRef> result = getDefValue<StringInit>("defaultValue");
  return result && !result->empty() ? result : std::nullopt;
}

const Init *AttrOrTypeParameter::getDef() const { return def->getArg(index); }

std::optional<Constraint> AttrOrTypeParameter::getConstraint() const {
  if (const auto *param = dyn_cast<DefInit>(getDef()))
    if (param->getDef()->isSubClassOf("Constraint"))
      return Constraint(param->getDef());
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// AttributeSelfTypeParameter
//===----------------------------------------------------------------------===//

bool AttributeSelfTypeParameter::classof(const AttrOrTypeParameter *param) {
  const Init *paramDef = param->getDef();
  if (const auto *paramDefInit = dyn_cast<DefInit>(paramDef))
    return paramDefInit->getDef()->isSubClassOf("AttributeSelfTypeParameter");
  return false;
}
