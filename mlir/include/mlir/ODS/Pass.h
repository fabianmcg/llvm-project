//===- Pass.h - ODS pass classes --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ODS PassOption, PassStatistic, and Pass classes, which
// model MLIR pass definitions independently of LLVM TableGen.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ODS_PASS_H_
#define MLIR_ODS_PASS_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <optional>
#include <string>
#include <vector>

namespace mlir {
namespace ods {

//===----------------------------------------------------------------------===//
// PassOption
//===----------------------------------------------------------------------===//

/// Models a pass command-line option, storing all fields as plain C++ strings.
class PassOption {
public:
  PassOption(StringRef cppVariableName, StringRef argument, StringRef type,
             std::optional<StringRef> defaultValue, StringRef description,
             std::optional<StringRef> additionalFlags, bool listOption)
      : cppVariableName(cppVariableName.str()), argument(argument.str()),
        type(type.str()),
        defaultValue(defaultValue
                         ? std::optional<std::string>(defaultValue->str())
                         : std::nullopt),
        description(description.str()),
        additionalFlags(additionalFlags
                            ? std::optional<std::string>(additionalFlags->str())
                            : std::nullopt),
        listOption(listOption) {}

  /// Returns the name for the C++ option variable.
  StringRef getCppVariableName() const { return cppVariableName; }

  /// Returns the command line argument to use for this option.
  StringRef getArgument() const { return argument; }

  /// Returns the C++ type of the option.
  StringRef getType() const { return type; }

  /// Returns the default value of the option, if any.
  std::optional<StringRef> getDefaultValue() const {
    if (!defaultValue)
      return std::nullopt;
    return StringRef(*defaultValue);
  }

  /// Returns the description for this option.
  StringRef getDescription() const { return description; }

  /// Returns the additional flags passed to the option constructor, if any.
  std::optional<StringRef> getAdditionalFlags() const {
    if (!additionalFlags)
      return std::nullopt;
    return StringRef(*additionalFlags);
  }

  /// Returns true if this is a list option.
  bool isListOption() const { return listOption; }

private:
  std::string cppVariableName;
  std::string argument;
  std::string type;
  std::optional<std::string> defaultValue;
  std::string description;
  std::optional<std::string> additionalFlags;
  bool listOption;
};

//===----------------------------------------------------------------------===//
// PassStatistic
//===----------------------------------------------------------------------===//

/// Models a pass statistic, storing all fields as plain C++ strings.
class PassStatistic {
public:
  PassStatistic(StringRef cppVariableName, StringRef name,
                StringRef description)
      : cppVariableName(cppVariableName.str()), name(name.str()),
        description(description.str()) {}

  /// Returns the name for the C++ statistic variable.
  StringRef getCppVariableName() const { return cppVariableName; }

  /// Returns the name of the statistic.
  StringRef getName() const { return name; }

  /// Returns the description for this statistic.
  StringRef getDescription() const { return description; }

private:
  std::string cppVariableName;
  std::string name;
  std::string description;
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

/// Models a pass definition, storing all fields as plain C++ strings and
/// collections with no dependency on LLVM TableGen types.
class Pass {
public:
  /// Returns the command line argument of the pass.
  StringRef getArgument() const { return argument; }

  /// Returns the name for the C++ base class.
  StringRef getBaseClass() const { return baseClass; }

  /// Returns the short 1-line summary of the pass.
  StringRef getSummary() const { return summary; }

  /// Returns the description of the pass.
  StringRef getDescription() const { return description; }

  /// Returns the C++ constructor call to create an instance of this pass.
  StringRef getConstructor() const { return constructor; }

  /// Returns the dialects this pass needs to be registered.
  ArrayRef<std::string> getDependentDialects() const {
    return dependentDialects;
  }

  /// Returns the options provided by this pass.
  ArrayRef<PassOption> getOptions() const { return options; }

  /// Returns the statistics provided by this pass.
  ArrayRef<PassStatistic> getStatistics() const { return statistics; }

protected:
  std::string argument;
  std::string baseClass;
  std::string summary;
  std::string description;
  std::string constructor;
  std::vector<std::string> dependentDialects;
  std::vector<PassOption> options;
  std::vector<PassStatistic> statistics;
};

} // namespace ods
} // namespace mlir

#endif // MLIR_ODS_PASS_H_
