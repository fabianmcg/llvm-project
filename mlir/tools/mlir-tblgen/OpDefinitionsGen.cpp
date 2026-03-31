//===- OpDefinitionsGen.cpp - MLIR op definitions generator ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/CppGen/OpDefinitionsGen.h"
#include "FormatGen.h"
#include "OpGenHelpers.h"
#include "mlir/TableGen/GenInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using llvm::raw_ostream;
using llvm::RecordKeeper;
using mlir::tblgen::formatErrorIsFatal;

static mlir::GenRegistration
    genOpDecls("gen-op-decls", "Generate op declarations",
               [](const RecordKeeper &records, raw_ostream &os) {
                 std::vector<const llvm::Record *> defs =
                     tblgen::getRequestedOpDefinitions(records);
                 SmallVector<ArrayRef<const llvm::Record *>, 4> shardedDefs;
                 tblgen::shardOpDefinitions(defs, shardedDefs);
                 return tblgen::emitOpDecls(records, defs, shardedDefs.size(),
                                            os, formatErrorIsFatal);
               });

static mlir::GenRegistration
    genOpDefs("gen-op-defs", "Generate op definitions",
              [](const RecordKeeper &records, raw_ostream &os) {
                std::vector<const llvm::Record *> defs =
                    tblgen::getRequestedOpDefinitions(records);
                SmallVector<ArrayRef<const llvm::Record *>, 4> shardedDefs;
                tblgen::shardOpDefinitions(defs, shardedDefs);
                return tblgen::emitOpDefs(records, defs, shardedDefs.size(),
                                          os, formatErrorIsFatal);
              });
