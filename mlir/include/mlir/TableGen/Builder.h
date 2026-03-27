//===- Builder.h - TableGen builder definitions -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Free functions for constructing mlir::ods::Builder from llvm::Record.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TABLEGEN_BUILDER_H_
#define MLIR_TABLEGEN_BUILDER_H_

#include "mlir/ODS/Builder.h"
#include "mlir/Support/LLVM.h"

namespace llvm {
class Record;
class SMLoc;
} // namespace llvm

namespace mlir {
namespace tblgen {

// Re-export the ODS type so existing callers using tblgen::Builder and
// tblgen::Builder::Parameter continue to work unchanged.
using Builder = mlir::ods::Builder;

/// Constructs an ods::Builder for an op builder by reading the common fields
/// (dagParams, body, odsCppDeprecated) from the TableGen record.
ods::Builder builderFromRecord(const llvm::Record *record,
                               ArrayRef<SMLoc> loc);

/// Constructs an ods::Builder for an AttrOrTypeDef builder, reading the common
/// fields plus the attr/type-specific returnType and hasInferredContextParam
/// fields.
ods::Builder attrOrTypeBuilderFromRecord(const llvm::Record *record,
                                         ArrayRef<SMLoc> loc);

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_BUILDER_H_
