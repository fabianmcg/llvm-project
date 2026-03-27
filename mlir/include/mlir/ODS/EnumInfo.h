//===- EnumInfo.h - ODS EnumInfo model --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pure C++ model classes for MLIR ODS enum definitions. No dependency on
// LLVM TableGen. The mlir::tblgen::EnumCase and mlir::tblgen::EnumInfo
// subclasses populate fields eagerly from llvm::Record objects.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ODS_ENUMINFO_H_
#define MLIR_ODS_ENUMINFO_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mlir {
namespace ods {

/// Pure C++ model for a single enum case. Fields are populated eagerly by
/// the tblgen::EnumCase constructor.
class EnumCase {
public:
  EnumCase() = default;
  EnumCase(StringRef symbol, StringRef str, int64_t value)
      : symbol(symbol.str()), str(str.str()), value(value) {}

  // Returns the symbol of this enum case.
  StringRef getSymbol() const { return symbol; }

  // Returns the textual representation of this enum case.
  StringRef getStr() const { return str; }

  // Returns the integer value of this enum case.
  int64_t getValue() const { return value; }

protected:
  std::string symbol;
  std::string str;
  int64_t value{0};
};

/// Pure C++ model for an MLIR ODS enum definition. Fields are populated
/// eagerly by the tblgen::EnumInfo constructor.
class EnumInfo {
public:
  EnumInfo() = default;

  // Returns true if this enum is an EnumAttrInfo (defines an attribute).
  bool isEnumAttr() const { return enumAttr; }

  // Returns true if this is a bit enum.
  bool isBitEnum() const { return bitEnum; }

  // Returns the C++ class name for this enum.
  StringRef getEnumClassName() const { return enumClassName; }

  // Returns the C++ namespace for this enum.
  StringRef getCppNamespace() const { return cppNamespace; }

  // Returns the summary of the enum.
  StringRef getSummary() const { return summary; }

  // Returns the description of the enum.
  StringRef getDescription() const { return description; }

  // Returns the bitwidth of the enum.
  int64_t getBitwidth() const { return bitwidth; }

  // Returns the underlying C++ type.
  StringRef getUnderlyingType() const { return underlyingType; }

  // Returns the name of the function converting the underlying value to symbol.
  StringRef getUnderlyingToSymbolFnName() const {
    return underlyingToSymbolFnName;
  }

  // Returns the name of the function converting a string to the symbol.
  StringRef getStringToSymbolFnName() const { return stringToSymbolFnName; }

  // Returns the name of the function converting a symbol to string.
  StringRef getSymbolToStringFnName() const { return symbolToStringFnName; }

  // Returns the return type of the symbol-to-string function.
  StringRef getSymbolToStringFnRetType() const {
    return symbolToStringFnRetType;
  }

  // Returns the name of the function returning the maximum enum value.
  StringRef getMaxEnumValFnName() const { return maxEnumValFnName; }

  // Returns the enum cases. Only valid fields (symbol, str, value) are
  // accessible here; use tblgen::EnumInfo::getAllCases() for the full
  // tblgen::EnumCase objects that also expose getDef().
  ArrayRef<ods::EnumCase> getCases() const { return cases; }

  // Returns whether a specialized attribute class should be generated.
  bool genSpecializedAttr() const { return specializedAttr; }

  // Returns the specialized attribute class name (only for enum attributes).
  StringRef getSpecializedAttrClassName() const {
    return specializedAttrClassName;
  }

  // Returns whether primary groups should be printed for bit enums.
  bool printBitEnumPrimaryGroups() const { return bitEnumPrimaryGroups; }

  // Returns whether bit enum values should be quoted.
  bool printBitEnumQuoted() const { return bitEnumQuoted; }

protected:
  std::string enumClassName;
  std::string cppNamespace;
  std::string summary;
  std::string description;
  std::string underlyingType;
  std::string underlyingToSymbolFnName;
  std::string stringToSymbolFnName;
  std::string symbolToStringFnName;
  std::string symbolToStringFnRetType;
  std::string maxEnumValFnName;
  std::string specializedAttrClassName;
  std::vector<ods::EnumCase> cases;
  int64_t bitwidth{0};
  bool enumAttr{false};
  bool bitEnum{false};
  bool specializedAttr{false};
  bool bitEnumPrimaryGroups{false};
  bool bitEnumQuoted{false};
};

} // namespace ods
} // namespace mlir

#endif // MLIR_ODS_ENUMINFO_H_
