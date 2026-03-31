//===- AttrOrTypeDefGen.h - AttrDef/TypeDef code generator ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares classes and functions for generating C++ definitions and
// declarations for MLIR attribute and type definitions from TableGen records.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_CPPGEN_ATTRORTYPEDEFGEN_H
#define MLIR_TABLEGEN_CPPGEN_ATTRORTYPEDEFGEN_H

#include "mlir/TableGen/AttrOrTypeDef.h"
#include "mlir/TableGen/Class.h"
#include "mlir/TableGen/Interfaces.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>
#include <vector>

namespace llvm {
class Record;
class RecordKeeper;
} // namespace llvm

namespace mlir {
namespace tblgen {

//===----------------------------------------------------------------------===//
// AttrOrTypeDefEmitter
//===----------------------------------------------------------------------===//

/// Generates C++ class declarations and definitions for a single
/// attribute or type definition derived from an AttrOrTypeDef TableGen record.
/// Previously named \c DefGen; renamed for clarity.
class AttrOrTypeDefEmitter {
public:
  /// Create the attribute or type class. If \p fatalOnError is true, assembly
  /// format parse failures are reported as fatal errors.
  AttrOrTypeDefEmitter(const AttrOrTypeDef &def, bool fatalOnError = true);

  void emitDecl(llvm::raw_ostream &os) const {
    if (storageCls && def.genStorageClass()) {
      llvm::NamespaceEmitter ns(os, def.getStorageNamespace());
      os << "struct " << def.getStorageClassName() << ";\n";
    }
    defCls.writeDeclTo(os);
  }
  void emitDef(llvm::raw_ostream &os) const {
    if (storageCls && def.genStorageClass()) {
      llvm::NamespaceEmitter ns(os, def.getStorageNamespace());
      storageCls->writeDeclTo(os); // everything is inline
    }
    defCls.writeDefTo(os);
  }

protected:
  /// Add traits from the TableGen definition to the class.
  void createParentWithTraits();
  /// Emit top-level declarations: using declarations and any extra class
  /// declarations.
  void emitTopLevelDeclarations();
  /// Emit the function that returns the type or attribute name.
  void emitName();
  /// Emit the dialect name as a static member variable.
  void emitDialectName();
  /// Emit attribute or type builders.
  void emitBuilders();
  /// Emit a verifier declaration for custom verification (impl. provided by
  /// the users).
  void emitVerifierDecl();
  /// Emit a verifier that checks type constraints.
  void emitInvariantsVerifierImpl();
  /// Emit an entry point for verification that calls the invariants and
  /// custom verifier.
  void emitInvariantsVerifier(bool hasImpl, bool hasCustomVerifier);
  /// Emit parsers and printers.
  void emitParserPrinter();
  /// Emit parameter accessors, if required.
  void emitAccessors();
  /// Emit interface methods.
  void emitInterfaceMethods();

  //===--------------------------------------------------------------------===//
  // Builder Emission

  /// Emit the default builder `Attribute::get`.
  void emitDefaultBuilder();
  /// Emit the checked builder `Attribute::getChecked`.
  void emitCheckedBuilder();
  /// Emit a custom builder.
  void emitCustomBuilder(const AttrOrTypeBuilder &builder);
  /// Emit a checked custom builder.
  void emitCheckedCustomBuilder(const AttrOrTypeBuilder &builder);

  //===--------------------------------------------------------------------===//
  // Interface Method Emission

  /// Emit methods for a trait.
  void emitTraitMethods(const InterfaceTrait &trait);
  /// Emit a trait method.
  void emitTraitMethod(const InterfaceMethod &method);
  /// Generate a using declaration for a trait method.
  void genTraitMethodUsingDecl(const InterfaceTrait &trait,
                               const InterfaceMethod &method);

  //===--------------------------------------------------------------------===//
  // OpAsm{Type,Attr}Interface Default Method Emission

  /// Emit 'getAlias' method using mnemonic as alias.
  void emitMnemonicAliasMethod();

  //===--------------------------------------------------------------------===//
  // Storage Class Emission
  void emitStorageClass();
  /// Generate the storage class constructor.
  void emitStorageConstructor();
  /// Emit the key type `KeyTy`.
  void emitKeyType();
  /// Emit the equality comparison operator.
  void emitEquals();
  /// Emit the key hash function.
  void emitHashKey();
  /// Emit the function to construct the storage class.
  void emitConstruct();

  //===--------------------------------------------------------------------===//
  // Utility Function Declarations

  /// Get the method parameters for a def builder, where the first several
  /// parameters may be different.
  SmallVector<MethodParameter>
  getBuilderParams(std::initializer_list<MethodParameter> prefix) const;

  //===--------------------------------------------------------------------===//
  // Class fields

  /// The attribute or type definition.
  const AttrOrTypeDef &def;
  /// The list of attribute or type parameters.
  ArrayRef<AttrOrTypeParameter> params;
  /// The attribute or type class.
  Class defCls;
  /// An optional attribute or type storage class. The storage class will
  /// exist if and only if the def has more than zero parameters.
  std::optional<Class> storageCls;

  /// The C++ base value of the def, either "Attribute" or "Type".
  StringRef valueType;
  /// The prefix/suffix of the TableGen def name, either "Attr" or "Type".
  StringRef defType;

  /// The set of using declarations for trait methods.
  llvm::StringSet<> interfaceUsingNames;

  /// Whether assembly format parse failures are fatal errors.
  bool fatalOnError;
};

//===----------------------------------------------------------------------===//
// DefGenerator
//===----------------------------------------------------------------------===//

/// Base generator for processing TableGen attr/type definitions.
class DefGenerator {
public:
  bool emitDecls(llvm::StringRef selectedDialect);
  bool emitDefs(llvm::StringRef selectedDialect);

protected:
  DefGenerator(llvm::ArrayRef<const llvm::Record *> defs, llvm::raw_ostream &os,
               llvm::StringRef defType, llvm::StringRef valueType,
               bool isAttrGenerator, bool fatalOnError = true)
      : defRecords(defs.begin(), defs.end()), os(os), defType(defType),
        valueType(valueType), isAttrGenerator(isAttrGenerator),
        fatalOnError(fatalOnError) {
    // Sort by occurrence in file.
    llvm::sort(defRecords, [](const llvm::Record *lhs,
                              const llvm::Record *rhs) {
      return lhs->getID() < rhs->getID();
    });
  }

  /// Emit the list of def type names.
  void emitTypeDefList(llvm::ArrayRef<AttrOrTypeDef> defs);
  /// Emit the code to dispatch between different defs during parsing/printing.
  void emitParsePrintDispatch(llvm::ArrayRef<AttrOrTypeDef> defs);

  /// The set of def records to emit.
  std::vector<const llvm::Record *> defRecords;
  /// The stream to emit to.
  llvm::raw_ostream &os;
  /// The prefix of the tablegen def name, e.g. Attr or Type.
  llvm::StringRef defType;
  /// The C++ base value type of the def, e.g. Attribute or Type.
  llvm::StringRef valueType;
  /// Flag indicating if this generator is for Attributes. False if the
  /// generator is for types.
  bool isAttrGenerator;
  /// Whether assembly format parse failures are fatal errors.
  bool fatalOnError;
};

/// A specialized generator for AttrDefs.
struct AttrDefGenerator : public DefGenerator {
  AttrDefGenerator(const llvm::RecordKeeper &records, llvm::raw_ostream &os,
                   bool fatalOnError = true);
};

/// A specialized generator for TypeDefs.
struct TypeDefGenerator : public DefGenerator {
  TypeDefGenerator(const llvm::RecordKeeper &records, llvm::raw_ostream &os,
                   bool fatalOnError = true);
};

//===----------------------------------------------------------------------===//
// Constraint Functions
//===----------------------------------------------------------------------===//

/// Emit declarations for all type constraints in \p records that have a C++
/// function name set.
void emitTypeConstraintDecls(const llvm::RecordKeeper &records,
                             llvm::raw_ostream &os);

/// Emit declarations for all attribute constraints in \p records that have a
/// C++ function name set.
void emitAttrConstraintDecls(const llvm::RecordKeeper &records,
                             llvm::raw_ostream &os);

/// Emit definitions for all type constraints in \p records that have a C++
/// function name set.
void emitTypeConstraintDefs(const llvm::RecordKeeper &records,
                            llvm::raw_ostream &os);

/// Emit definitions for all attribute constraints in \p records that have a
/// C++ function name set.
void emitAttrConstraintDefs(const llvm::RecordKeeper &records,
                            llvm::raw_ostream &os);

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_CPPGEN_ATTRORTYPEDEFGEN_H
