//===- AttrOrTypeDef.h - ODS AttrOrTypeDef model ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pure C++ model class for MLIR ODS attribute and type definitions. No
// dependency on LLVM TableGen. The mlir::tblgen::AttrOrTypeDef subclass
// populates the fields eagerly from an llvm::Record during construction.
//
// Builders, traits, and parameters are not stored here; they require
// TableGen types and are accessed through the tblgen subclass.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ODS_ATTRORTYPEDEF_H_
#define MLIR_ODS_ATTRORTYPEDEF_H_

#include "mlir/ODS/Dialect.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"

#include <optional>
#include <string>

namespace mlir {
namespace ods {

/// Pure C++ model for an MLIR ODS attribute or type definition. All scalar
/// fields are populated eagerly by the tblgen::AttrOrTypeDef constructor and
/// are thereafter immutable.
///
/// Builders, traits, and parameters are not stored in this base class because
/// they depend on TableGen types (llvm::Record, llvm::DagInit, etc.). They are
/// accessible through the tblgen subclass.
class AttrOrTypeDef {
public:
  AttrOrTypeDef() = default;

  /// Returns the dialect this definition belongs to.
  const ods::Dialect &getDialect() const { return dialect; }

  /// Returns the TableGen record name of this definition.
  StringRef getName() const { return name; }

  /// Returns the C++ class name to generate.
  StringRef getCppClassName() const { return cppClassName; }

  /// Returns the C++ base class to use when generating this definition.
  StringRef getCppBaseClassName() const { return cppBaseClassName; }

  /// Returns true if this definition has a description.
  bool hasDescription() const { return hasDescriptionFlag; }

  /// Returns the description of this definition.
  StringRef getDescription() const { return description; }

  /// Returns true if this definition has a summary.
  bool hasSummary() const { return hasSummaryFlag; }

  /// Returns the summary of this definition.
  StringRef getSummary() const { return summary; }

  /// Returns the name of the storage class for this definition.
  StringRef getStorageClassName() const { return storageClassName; }

  /// Returns the C++ namespace for this definition's storage class.
  StringRef getStorageNamespace() const { return storageNamespace; }

  /// Returns true if the storage class should be generated.
  bool genStorageClass() const { return genStorageClassFlag; }

  /// Returns true if a custom storage class constructor is present.
  bool hasStorageCustomConstructor() const {
    return hasStorageCustomConstructorFlag;
  }

  /// Returns the mnemonic used in printer/parser methods, if set.
  std::optional<StringRef> getMnemonic() const {
    if (!mnemonic)
      return std::nullopt;
    return StringRef(*mnemonic);
  }

  /// Returns true if this definition has a custom assembly format in C++.
  bool hasCustomAssemblyFormat() const { return hasCustomAssemblyFormatFlag; }

  /// Returns the custom assembly format string, if one was specified.
  std::optional<StringRef> getAssemblyFormat() const {
    if (!assemblyFormat)
      return std::nullopt;
    return StringRef(*assemblyFormat);
  }

  /// Returns true if accessors based on the parameters should be generated.
  bool genAccessors() const { return genAccessorsFlag; }

  /// Returns true if verify declaration and getChecked method should be
  /// generated.
  bool genVerifyDecl() const { return genVerifyDeclFlag; }

  /// Returns true if type constraint verification and getChecked should be
  /// generated. This is precomputed at construction time.
  bool genVerifyInvariantsImpl() const { return verifyInvariantsImplFlag; }

  /// Returns the extra class declaration code, if any.
  std::optional<StringRef> getExtraDecls() const {
    if (!extraDecls)
      return std::nullopt;
    return StringRef(*extraDecls);
  }

  /// Returns the extra class definition code, if any.
  std::optional<StringRef> getExtraDefs() const {
    if (!extraDefs)
      return std::nullopt;
    return StringRef(*extraDefs);
  }

  /// Returns true if a default 'getAlias' implementation using the mnemonic
  /// should be generated.
  bool genMnemonicAlias() const { return genMnemonicAliasFlag; }

  /// Returns true if the default get/getChecked methods should be skipped
  /// during generation.
  bool skipDefaultBuilders() const { return skipDefaultBuildersFlag; }

protected:
  ods::Dialect dialect;
  std::string name;
  std::string cppClassName;
  std::string cppBaseClassName;
  std::string description;
  std::string summary;
  std::string storageClassName;
  std::string storageNamespace;
  std::optional<std::string> mnemonic;
  std::optional<std::string> assemblyFormat;
  std::optional<std::string> extraDecls;
  std::optional<std::string> extraDefs;
  bool hasDescriptionFlag{false};
  bool hasSummaryFlag{false};
  bool genStorageClassFlag{false};
  bool hasStorageCustomConstructorFlag{false};
  bool hasCustomAssemblyFormatFlag{false};
  bool genAccessorsFlag{false};
  bool genVerifyDeclFlag{false};
  bool verifyInvariantsImplFlag{false};
  bool genMnemonicAliasFlag{false};
  bool skipDefaultBuildersFlag{false};
};

} // namespace ods
} // namespace mlir

#endif // MLIR_ODS_ATTRORTYPEDEF_H_
