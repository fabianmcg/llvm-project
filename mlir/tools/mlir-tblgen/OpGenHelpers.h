//===- OpGenHelpers.h - MLIR operation generator helpers --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRTBLGEN_OPGENHELPERS_H_
#define MLIR_TOOLS_MLIRTBLGEN_OPGENHELPERS_H_

#include "mlir/TableGen/CppGen/OpGenHelpers.h"

namespace mlir {
namespace tblgen {

/// Returns all op definitions filtered by the "op-include-regex" and
/// "op-exclude-regex" command-line options.
std::vector<const llvm::Record *>
getRequestedOpDefinitions(const llvm::RecordKeeper &records);

/// Shard op definitions using the "op-shard-count" command-line option.
void shardOpDefinitions(
    ArrayRef<const llvm::Record *> defs,
    SmallVectorImpl<ArrayRef<const llvm::Record *>> &shardedDefs);

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TOOLS_MLIRTBLGEN_OPGENHELPERS_H_
