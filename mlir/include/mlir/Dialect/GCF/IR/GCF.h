//===- GCFOps.h - Generic Control Flow --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines generic control flow operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_GCF_GCF_H
#define MLIR_DIALECT_GCF_GCF_H

#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/GCF/IR/GCFOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/GCF/IR/GCFOps.h.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/GCF/IR/GCFOpsAttributes.h.inc"

#endif // MLIR_DIALECT_GCF_GCF_H
