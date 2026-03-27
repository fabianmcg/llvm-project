//===- EnumInfo.h - EnumInfo wrapper class --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// EnumInfo wrapper to simplify using a TableGen Record defining an Enum
// via EnumInfo and its `EnumCase`s.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_ENUMINFO_H_
#define MLIR_TABLEGEN_ENUMINFO_H_

#include "mlir/ODS/EnumInfo.h"
#include "mlir/Support/LLVM.h"
#include "mlir/TableGen/Attribute.h"
#include "llvm/ADT/StringRef.h"

namespace llvm {
class DefInit;
class Record;
} // namespace llvm

namespace mlir::tblgen {

// Wrapper class providing around enum cases defined in TableGen.
// Derives from ods::EnumCase and adds getDef() for TableGen-specific access.
class EnumCase : public mlir::ods::EnumCase {
public:
  explicit EnumCase(const llvm::Record *record);
  explicit EnumCase(const llvm::DefInit *init);

  // Returns the symbol of this enum attribute case.
  StringRef getSymbol() const { return mlir::ods::EnumCase::getSymbol(); }

  // Returns the textual representation of this enum attribute case.
  StringRef getStr() const { return mlir::ods::EnumCase::getStr(); }

  // Returns the value of this enum attribute case.
  int64_t getValue() const { return mlir::ods::EnumCase::getValue(); }

  // Returns the TableGen definition this EnumAttrCase was constructed from.
  const llvm::Record &getDef() const;

protected:
  // The TableGen definition of this enum case.
  const llvm::Record *def;
};

// Wrapper class providing helper methods for accessing enums defined
// in TableGen using EnumInfo. Derives from ods::EnumInfo and adds
// TableGen-specific access methods.
class EnumInfo : public mlir::ods::EnumInfo {
public:
  explicit EnumInfo(const llvm::Record *record);
  explicit EnumInfo(const llvm::Record &record);
  explicit EnumInfo(const llvm::DefInit *init);

  // Returns true if the given EnumInfo is a subclass of the named TableGen
  // class.
  bool isSubClassOf(StringRef className) const;

  // Returns true if this enum is an EnumAttrInfo, thus making it define an
  // attribute.
  bool isEnumAttr() const { return mlir::ods::EnumInfo::isEnumAttr(); }

  // Create the `Attribute` wrapper around this EnumInfo if it is defining an
  // attribute.
  std::optional<Attribute> asEnumAttr() const;

  // Returns true if this is a bit enum.
  bool isBitEnum() const { return mlir::ods::EnumInfo::isBitEnum(); }

  // Returns the enum class name.
  StringRef getEnumClassName() const {
    return mlir::ods::EnumInfo::getEnumClassName();
  }

  // Returns the C++ namespaces this enum class should be placed in.
  StringRef getCppNamespace() const {
    return mlir::ods::EnumInfo::getCppNamespace();
  }

  // Returns the summary of the enum.
  StringRef getSummary() const { return mlir::ods::EnumInfo::getSummary(); }

  // Returns the description of the enum.
  StringRef getDescription() const {
    return mlir::ods::EnumInfo::getDescription();
  }

  // Returns the bitwidth of the enum.
  int64_t getBitwidth() const { return mlir::ods::EnumInfo::getBitwidth(); }

  // Returns the underlying type.
  StringRef getUnderlyingType() const {
    return mlir::ods::EnumInfo::getUnderlyingType();
  }

  // Returns the name of the utility function that converts a value of the
  // underlying type to the corresponding symbol.
  StringRef getUnderlyingToSymbolFnName() const {
    return mlir::ods::EnumInfo::getUnderlyingToSymbolFnName();
  }

  // Returns the name of the utility function that converts a string to the
  // corresponding symbol.
  StringRef getStringToSymbolFnName() const {
    return mlir::ods::EnumInfo::getStringToSymbolFnName();
  }

  // Returns the name of the utility function that converts a symbol to the
  // corresponding string.
  StringRef getSymbolToStringFnName() const {
    return mlir::ods::EnumInfo::getSymbolToStringFnName();
  }

  // Returns the return type of the utility function that converts a symbol to
  // the corresponding string.
  StringRef getSymbolToStringFnRetType() const {
    return mlir::ods::EnumInfo::getSymbolToStringFnRetType();
  }

  // Returns the name of the utility function that returns the max enum value
  // used within the enum class.
  StringRef getMaxEnumValFnName() const {
    return mlir::ods::EnumInfo::getMaxEnumValFnName();
  }

  // Returns all allowed cases for this enum attribute as tblgen::EnumCase
  // objects (which expose getDef() for TableGen-specific access).
  std::vector<EnumCase> getAllCases() const;

  // Only applicable for enum attributes.

  bool genSpecializedAttr() const {
    return mlir::ods::EnumInfo::genSpecializedAttr();
  }
  const llvm::Record *getBaseAttrClass() const;
  StringRef getSpecializedAttrClassName() const {
    return mlir::ods::EnumInfo::getSpecializedAttrClassName();
  }

  // Only applicable for bit enums.

  bool printBitEnumPrimaryGroups() const {
    return mlir::ods::EnumInfo::printBitEnumPrimaryGroups();
  }
  bool printBitEnumQuoted() const {
    return mlir::ods::EnumInfo::printBitEnumQuoted();
  }

  // Returns the TableGen definition this EnumAttrCase was constructed from.
  const llvm::Record &getDef() const;

protected:
  // The TableGen definition of this constraint.
  const llvm::Record *def;
};

} // namespace mlir::tblgen

#endif
