//===-- mlir-c/Dialect/Ptr.h - C API for Ptr ----------------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_DIALECT_PTR_H
#define MLIR_C_DIALECT_PTR_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Ptr, ptr);

/// Creates a !ptr.ptr type.
MLIR_CAPI_EXPORTED MlirType mlirPtrTypeGet(MlirContext ctx,
                                           unsigned addressSpace);

/// Creates a !ptr.ptr type.
MLIR_CAPI_EXPORTED MlirType mlirPtrTypeGetAttr(MlirContext ctx,
                                               MlirAttribute addressSpace);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_DIALECT_PTR_H
