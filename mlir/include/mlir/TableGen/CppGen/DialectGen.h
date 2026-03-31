//===- DialectGen.h - MLIR dialect C++ generation utilities -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares utilities for generating C++ definitions for dialects from
// TableGen records.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_CPPGEN_DIALECTGEN_H
#define MLIR_TABLEGEN_CPPGEN_DIALECTGEN_H

#include "mlir/TableGen/Dialect.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/TableGen/Record.h"
#include <optional>
#include <string>
#include <utility>

namespace llvm {
class RecordKeeper;
class raw_ostream;
class DagInit;
} // namespace llvm

namespace mlir {
namespace tblgen {

/// Iterator type for filtering records by dialect.
using DialectFilterIterator =
    llvm::filter_iterator<llvm::ArrayRef<llvm::Record *>::iterator,
                          std::function<bool(const llvm::Record *)>>;

/// Given a set of records for T, return the subset whose dialect matches
/// \p dialect.
template <typename T>
llvm::iterator_range<DialectFilterIterator>
filterForDialect(llvm::ArrayRef<llvm::Record *> records, Dialect &dialect) {
  auto filterFn = [&](const llvm::Record *record) {
    return T(record).getDialect() == dialect;
  };
  return {DialectFilterIterator(records.begin(), records.end(), filterFn),
          DialectFilterIterator(records.end(), records.end(), filterFn)};
}

/// Populate \p discardableAttributes from the given DagInit of discardable
/// attribute descriptors.
void populateDiscardableAttributes(
    Dialect &dialect, const llvm::DagInit *discardableAttrDag,
    llvm::SmallVectorImpl<std::pair<std::string, std::string>>
        &discardableAttributes);

/// Find the dialect to generate from \p dialects. If \p selectedDialect is
/// empty, the dialect is auto-detected (succeeds only when exactly one dialect
/// is present). Returns std::nullopt and prints an error on failure.
std::optional<Dialect> findDialectToGenerate(llvm::ArrayRef<Dialect> dialects,
                                             llvm::StringRef selectedDialect);

/// Emit the C++ class declaration for \p dialect.
void emitDialectDecl(Dialect &dialect, llvm::raw_ostream &os);

/// Emit the C++ class declarations for all dialects in \p records, selecting
/// the one identified by \p selectedDialect.
bool emitDialectDecls(const llvm::RecordKeeper &records,
                      llvm::StringRef selectedDialect, llvm::raw_ostream &os);

/// Emit the C++ constructor and destructor definitions for \p dialect.
void emitDialectDef(Dialect &dialect, const llvm::RecordKeeper &records,
                    llvm::raw_ostream &os);

/// Emit the C++ constructor and destructor definitions for all dialects in
/// \p records, selecting the one identified by \p selectedDialect.
bool emitDialectDefs(const llvm::RecordKeeper &records,
                     llvm::StringRef selectedDialect, llvm::raw_ostream &os);

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_CPPGEN_DIALECTGEN_H
