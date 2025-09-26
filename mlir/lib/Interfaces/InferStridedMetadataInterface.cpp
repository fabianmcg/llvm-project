//===- InferIntRangeInterface.cpp -  Integer range inference interface ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/InferStridedMetadataInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include <optional>

using namespace mlir;

#include "mlir/Interfaces/InferStridedMetadataInterface.cpp.inc"

void StridedMetadataRange::print(raw_ostream &os) const {
  (void)0;
  return;
}
