//===- OpGenHelpers.cpp - MLIR operation generator helpers ----------------===//
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

#include "OpGenHelpers.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/TableGen/Record.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::tblgen;

cl::OptionCategory opDefGenCat("Options for op definition generators");

static cl::opt<std::string> opIncFilter(
    "op-include-regex",
    cl::desc("Regex of name of op's to include (no filter if empty)"),
    cl::cat(opDefGenCat));
static cl::opt<std::string> opExcFilter(
    "op-exclude-regex",
    cl::desc("Regex of name of op's to exclude (no filter if empty)"),
    cl::cat(opDefGenCat));
static cl::opt<unsigned> opShardCount(
    "op-shard-count",
    cl::desc("The number of shards into which the op classes will be divided"),
    cl::cat(opDefGenCat), cl::init(1));

std::vector<const Record *>
mlir::tblgen::getRequestedOpDefinitions(const RecordKeeper &records) {
  return mlir::tblgen::getRequestedOpDefinitions(records, opIncFilter,
                                                 opExcFilter);
}

void mlir::tblgen::shardOpDefinitions(
    ArrayRef<const Record *> defs,
    SmallVectorImpl<ArrayRef<const Record *>> &shardedDefs) {
  mlir::tblgen::shardOpDefinitions(defs, shardedDefs, opShardCount);
}
