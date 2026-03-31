//===- OpGenHelpers.h - MLIR operation generator helpers --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines helpers used in the op generators.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_CPPGEN_OPGENHELPERS_H
#define MLIR_TABLEGEN_CPPGEN_OPGENHELPERS_H

#include "mlir/Support/LLVM.h"
#include "llvm/TableGen/Record.h"
#include <vector>

namespace mlir {
namespace tblgen {

/// Returns all op definitions from \p records whose operation name matches the
/// optional include/exclude regex filters. Pass empty strings to skip
/// filtering.
std::vector<const llvm::Record *>
getRequestedOpDefinitions(const llvm::RecordKeeper &records,
                          llvm::StringRef includeRegex,
                          llvm::StringRef excludeRegex);

/// Checks whether \p str is a Python keyword or would shadow a builtin
/// function. Regenerate using:
///   python -c"print(set(sorted(__import__('keyword').kwlist)))"
bool isPythonReserved(llvm::StringRef str);

/// Shard \p defs into \p shardCount approximately equal-sized shards and
/// append them to \p shardedDefs.
void shardOpDefinitions(
    ArrayRef<const llvm::Record *> defs,
    SmallVectorImpl<ArrayRef<const llvm::Record *>> &shardedDefs,
    unsigned shardCount);

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_CPPGEN_OPGENHELPERS_H
