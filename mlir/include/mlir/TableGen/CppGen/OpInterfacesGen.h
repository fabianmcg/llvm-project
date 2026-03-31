//===- OpInterfacesGen.h - Op/Attr/Type interface generator -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_CPPGEN_OPINTERFACESGEN_H
#define MLIR_TABLEGEN_CPPGEN_OPINTERFACESGEN_H

#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/Interfaces.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <vector>

namespace llvm {
class Record;
class RecordKeeper;
} // namespace llvm

namespace mlir {
namespace tblgen {

/// Get all interface definitions of the given kind, excluding those that
/// subclass "Declare<kind>InterfaceMethods".
std::vector<const llvm::Record *>
getAllInterfaceDefinitions(const llvm::RecordKeeper &records,
                           llvm::StringRef name);

//===----------------------------------------------------------------------===//
// InterfaceGenerator
//===----------------------------------------------------------------------===//

/// Base generator for processing TableGen interface definitions.
class InterfaceGenerator {
public:
  bool emitInterfaceDefs();
  bool emitInterfaceDecls();
  bool emitInterfaceDocs();

protected:
  InterfaceGenerator(std::vector<const llvm::Record *> &&defs,
                     llvm::raw_ostream &os)
      : defs(std::move(defs)), os(os) {}

  void emitConceptDecl(const Interface &interface);
  void emitModelDecl(const Interface &interface);
  void emitModelMethodsDef(const Interface &interface);
  void forwardDeclareInterface(const Interface &interface);
  void emitInterfaceDecl(const Interface &interface);
  void emitInterfaceTraitDecl(const Interface &interface);

  /// The set of interface records to emit.
  std::vector<const llvm::Record *> defs;
  /// The stream to emit to.
  llvm::raw_ostream &os;
  /// The C++ value type of the interface, e.g. Operation*.
  llvm::StringRef valueType;
  /// The C++ base interface type.
  llvm::StringRef interfaceBaseType;
  /// The name of the typename for the value template.
  llvm::StringRef valueTemplate;
  /// The name of the substitution variable for the value.
  llvm::StringRef substVar;
  /// The format contexts to use for methods.
  FmtContext nonStaticMethodFmt;
  FmtContext traitMethodFmt;
  FmtContext extraDeclsFmt;
};

/// A specialized generator for attribute interfaces.
struct AttrInterfaceGenerator : public InterfaceGenerator {
  AttrInterfaceGenerator(const llvm::RecordKeeper &records,
                         llvm::raw_ostream &os);
};

/// A specialized generator for operation interfaces.
struct OpInterfaceGenerator : public InterfaceGenerator {
  OpInterfaceGenerator(const llvm::RecordKeeper &records,
                       llvm::raw_ostream &os);
};

/// A specialized generator for type interfaces.
struct TypeInterfaceGenerator : public InterfaceGenerator {
  TypeInterfaceGenerator(const llvm::RecordKeeper &records,
                         llvm::raw_ostream &os);
};

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_CPPGEN_OPINTERFACESGEN_H
