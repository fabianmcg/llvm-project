//===- Ptr.cpp - C Interface for Ptr dialect ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/Ptr.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/Ptr/IR/PtrOps.h"

using namespace mlir;
using namespace mlir::ptr;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Ptr, ptr, PtrDialect)

MlirType mlirPtrTypeGet(MlirContext ctx, unsigned addressSpace) {
  return wrap(ptr::PtrType::get(unwrap(ctx), addressSpace));
}

MlirType mlirPtrTypeGetAttr(MlirContext ctx, MlirAttribute addressSpace) {
  return wrap(ptr::PtrType::get(unwrap(ctx), unwrap(addressSpace)));
}
