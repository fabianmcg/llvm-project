//===- OpDefinitionsGen.h - Op definitions generator -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_CPPGEN_OPDEFINITIONSGEN_H
#define MLIR_TABLEGEN_CPPGEN_OPDEFINITIONSGEN_H

#include "mlir/TableGen/Dialect.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
class Record;
class RecordKeeper;
} // namespace llvm

namespace mlir {
namespace tblgen {

/// Emit op declarations for all op records in \p defs. If \p fatalOnError is
/// true, assembly format parse errors are fatal; otherwise they are ignored.
bool emitOpDecls(const llvm::RecordKeeper &records,
                 llvm::ArrayRef<const llvm::Record *> defs,
                 unsigned shardCount, llvm::raw_ostream &os,
                 bool fatalOnError = true);

/// Generate the dialect op registration hook and op class definitions for a
/// shard of ops.
void emitOpDefShard(const llvm::RecordKeeper &records,
                    llvm::ArrayRef<const llvm::Record *> shardDefs,
                    const Dialect &dialect, unsigned shardIndex,
                    unsigned shardCount, llvm::raw_ostream &os,
                    bool fatalOnError = true);

/// Emit op definitions for all op records in \p defs. If \p fatalOnError is
/// true, assembly format parse errors are fatal; otherwise they are ignored.
bool emitOpDefs(const llvm::RecordKeeper &records,
                llvm::ArrayRef<const llvm::Record *> defs,
                unsigned shardCount, llvm::raw_ostream &os,
                bool fatalOnError = true);

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_CPPGEN_OPDEFINITIONSGEN_H
