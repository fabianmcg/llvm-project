//===- Builder.h - ODS Builder class ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ODS Builder class, which models an op/type/attr
// builder method independently of LLVM TableGen.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ODS_BUILDER_H_
#define MLIR_ODS_BUILDER_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <optional>
#include <string>

namespace mlir {
namespace ods {

/// Models a builder method, storing all data as plain C++ fields with no
/// dependency on LLVM TableGen types.
class Builder {
public:
  /// A single parameter in a builder method.
  class Parameter {
  public:
    /// Constructs a Parameter with an optional name, a required C++ type, and
    /// an optional default value.
    Parameter(std::optional<StringRef> name, StringRef cppType,
              std::optional<StringRef> defaultValue)
        : name(name ? std::optional<std::string>(name->str()) : std::nullopt),
          cppType(cppType.str()),
          defaultValue(defaultValue
                           ? std::optional<std::string>(defaultValue->str())
                           : std::nullopt) {}

    /// Returns an optional string containing the name of this parameter.
    std::optional<StringRef> getName() const {
      if (!name)
        return std::nullopt;
      return StringRef(*name);
    }

    /// Returns a string containing the C++ type of this parameter.
    StringRef getCppType() const { return cppType; }

    /// Returns an optional string containing the default value to use for this
    /// parameter.
    std::optional<StringRef> getDefaultValue() const {
      if (!defaultValue)
        return std::nullopt;
      return StringRef(*defaultValue);
    }

  private:
    std::optional<std::string> name;
    std::string cppType;
    std::optional<std::string> defaultValue;
  };

  /// Returns a list of parameters used in this build method.
  ArrayRef<Parameter> getParameters() const { return parameters; }

  /// Returns an optional string containing the body of the builder.
  std::optional<StringRef> getBody() const {
    if (!body)
      return std::nullopt;
    return StringRef(*body);
  }

  /// Returns the deprecation message, or std::nullopt if not deprecated.
  std::optional<StringRef> getDeprecatedMessage() const {
    if (!deprecatedMessage)
      return std::nullopt;
    return StringRef(*deprecatedMessage);
  }

  /// Returns the optional return type for attr/type builders (empty for op
  /// builders).
  std::optional<StringRef> getReturnType() const {
    if (!returnType)
      return std::nullopt;
    return StringRef(*returnType);
  }

  /// Returns true if this attr/type builder can infer the MLIRContext
  /// parameter.
  bool hasInferredContextParameter() const { return inferredContextParameter; }

// All fields are public so that free factory functions (e.g.,
// tblgen::builderFromRecord) can populate them without requiring friendship.
// Callers should use the accessor methods above for read access.
public:
  SmallVector<Parameter> parameters;
  std::optional<std::string> body;
  std::optional<std::string> deprecatedMessage;
  std::optional<std::string> returnType;
  bool inferredContextParameter{false};
};

} // namespace ods
} // namespace mlir

#endif // MLIR_ODS_BUILDER_H_
