//===- Dialect.h - ODS dialect class ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ODS Dialect class, which models a MLIR dialect
// independently of LLVM TableGen.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ODS_DIALECT_H_
#define MLIR_ODS_DIALECT_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <optional>
#include <string>
#include <vector>

namespace mlir {
namespace ods {

/// Contains a MLIR dialect's information and provides helper methods for
/// accessing it. This class is independent of LLVM TableGen; all data is
/// stored as explicit C++ fields.
class Dialect {
public:
  /// Constructs an undefined (null) dialect.
  Dialect() = default;

  // Returns whether the dialect is defined.
  explicit operator bool() const { return defined; }
  bool isDefined() const { return defined; }

  // Returns the name of this dialect.
  StringRef getName() const { return name; }

  // Returns the C++ namespaces that ops of this dialect should be placed into.
  StringRef getCppNamespace() const { return cppNamespace; }

  // Returns this dialect's C++ class name.
  StringRef getCppClassName() const { return cppClassName; }

  // Returns the summary description of the dialect.
  StringRef getSummary() const { return summary; }

  // Returns the description of the dialect.
  StringRef getDescription() const { return description; }

  // Returns the list of dependent dialect class names.
  ArrayRef<std::string> getDependentDialects() const {
    return dependentDialects;
  }

  // Returns the dialect's extra class declaration code, if any.
  std::optional<StringRef> getExtraClassDeclaration() const {
    if (!extraClassDeclaration)
      return std::nullopt;
    return StringRef(*extraClassDeclaration);
  }

  bool hasCanonicalizer() const { return canonicalizer; }
  bool hasConstantMaterializer() const { return constantMaterializer; }
  bool hasNonDefaultDestructor() const { return nonDefaultDestructor; }
  bool hasOperationAttrVerify() const { return operationAttrVerify; }
  bool hasRegionArgAttrVerify() const { return regionArgAttrVerify; }
  bool hasRegionResultAttrVerify() const { return regionResultAttrVerify; }
  bool hasOperationInterfaceFallback() const {
    return operationInterfaceFallback;
  }
  bool useDefaultAttributePrinterParser() const {
    return defaultAttributePrinterParser;
  }
  bool useDefaultTypePrinterParser() const { return defaultTypePrinterParser; }
  bool isExtensible() const { return extensible; }

  bool operator==(const Dialect &other) const { return name == other.name; }
  bool operator!=(const Dialect &other) const { return !(*this == other); }
  bool operator<(const Dialect &other) const { return name < other.name; }

protected:
  bool defined{false};

  std::string name;
  std::string cppNamespace;
  std::string cppClassName;
  std::string summary;
  std::string description;
  std::vector<std::string> dependentDialects;
  std::optional<std::string> extraClassDeclaration;

  bool canonicalizer{false};
  bool constantMaterializer{false};
  bool nonDefaultDestructor{false};
  bool operationAttrVerify{false};
  bool regionArgAttrVerify{false};
  bool regionResultAttrVerify{false};
  bool operationInterfaceFallback{false};
  bool defaultAttributePrinterParser{false};
  bool defaultTypePrinterParser{false};
  bool extensible{false};
};

} // namespace ods
} // namespace mlir

#endif // MLIR_ODS_DIALECT_H_
