//===- Operator.h - ODS Operator model --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pure C++ model class for an MLIR ODS operator (op) definition. No dependency
// on LLVM TableGen. The mlir::tblgen::Operator subclass populates the scalar
// fields eagerly from an llvm::Record during construction and adds accessors
// for TableGen-specific data structures (arguments, results, traits, etc.).
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ODS_OPERATOR_H_
#define MLIR_ODS_OPERATOR_H_

#include "mlir/ODS/Dialect.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"

#include <optional>
#include <string>

namespace mlir {
namespace ods {

/// Pure C++ model for an MLIR ODS operator definition. All scalar metadata
/// is stored as plain C++ fields and populated eagerly by the
/// tblgen::Operator constructor. TableGen-specific data (arguments, results,
/// traits, regions, successors, builders, type-inference mappings) remains in
/// the tblgen subclass.
class Operator {
public:
  Operator() = default;

  /// Returns the dialect name of this op.
  StringRef getDialectName() const { return dialectName; }

  /// Returns the unqualified C++ class name.
  StringRef getCppClassName() const { return cppClassName; }

  /// Returns the C++ namespace.
  StringRef getCppNamespace() const { return cppNamespace; }

  /// Returns the canonical operation name, e.g. "dialect.op_name".
  StringRef getOperationName() const { return operationName; }

  /// Returns the TableGen "opName" field value (without dialect prefix).
  StringRef getOpName() const { return opName; }

  /// Returns the name of the adaptor C++ class.
  std::string getAdaptorName() const {
    return std::string(getCppClassName()) + "Adaptor";
  }

  /// Returns the name of the generic adaptor C++ class.
  std::string getGenericAdaptorName() const {
    return std::string(getCppClassName()) + "GenericAdaptor";
  }

  /// Returns the C++ class name prefixed with namespaces.
  std::string getQualCppClassName() const {
    if (cppNamespace.empty())
      return cppClassName;
    return cppNamespace + "::" + cppClassName;
  }

  /// Returns true if this op has a description.
  bool hasDescription() const { return !description.empty(); }

  /// Returns the description of this op.
  StringRef getDescription() const { return description; }

  /// Returns true if this op has a summary.
  bool hasSummary() const { return !summary.empty(); }

  /// Returns the summary of this op.
  StringRef getSummary() const { return summary; }

  /// Returns true if this op has a declarative assembly format.
  bool hasAssemblyFormat() const { return assemblyFormat.has_value(); }

  /// Returns the declarative assembly format string. Only valid if
  /// hasAssemblyFormat() returns true.
  StringRef getAssemblyFormat() const {
    assert(assemblyFormat && "no assembly format");
    return StringRef(*assemblyFormat);
  }

  /// Returns this op's extra class declaration code, if any.
  StringRef getExtraClassDeclaration() const { return extraClassDeclaration; }

  /// Returns this op's extra class definition code, if any.
  StringRef getExtraClassDefinition() const { return extraClassDefinition; }

  /// Returns true if this op has a folder.
  bool hasFolder() const { return hasFolderFlag; }

  /// Returns true if this op uses custom properties encoding.
  bool useCustomPropertiesEncoding() const {
    return useCustomPropertiesEncodingFlag;
  }

  /// Returns true if default builders should not be generated.
  bool skipDefaultBuilders() const { return skipDefaultBuildersFlag; }

  /// Returns true if this op has variable length operands or results.
  bool isVariadic() const { return isVariadicFlag; }

  /// Returns true if the types of all results are known.
  bool allResultTypesKnown() const { return allResultsHaveKnownTypes; }

  /// Returns the getter method name for an accessor of `name`.
  std::string getGetterName(StringRef name) const {
    return "get" +
           llvm::convertToCamelFromSnakeCase(name, /*capitalizeFirst=*/true);
  }

  /// Returns the setter method name for an accessor of `name`.
  std::string getSetterName(StringRef name) const {
    return "set" +
           llvm::convertToCamelFromSnakeCase(name, /*capitalizeFirst=*/true);
  }

  /// Returns the remover method name for an accessor of `name`.
  std::string getRemoverName(StringRef name) const {
    return "remove" +
           llvm::convertToCamelFromSnakeCase(name, /*capitalizeFirst=*/true);
  }

protected:
  std::string dialectName;
  std::string cppClassName;
  std::string cppNamespace;
  std::string operationName;
  std::string opName;
  std::string description;
  std::string summary;
  std::optional<std::string> assemblyFormat;
  std::string extraClassDeclaration;
  std::string extraClassDefinition;
  bool hasFolderFlag{false};
  bool useCustomPropertiesEncodingFlag{false};
  bool skipDefaultBuildersFlag{false};
  bool isVariadicFlag{false};
  bool allResultsHaveKnownTypes{false};
};

} // namespace ods
} // namespace mlir

#endif // MLIR_ODS_OPERATOR_H_
