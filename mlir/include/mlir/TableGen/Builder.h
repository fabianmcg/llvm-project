//===- Builder.h - Builder classes ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TableGen wrapper around ODS Builder. Derives from mlir::ods::Builder and
// populates the ODS fields from an llvm::Record in its constructor.
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

/// TableGen wrapper for a builder method. Derives from mlir::ods::Builder and
/// additionally stores the underlying llvm::Record. All ODS fields are
/// populated eagerly in the constructor.
class Builder : public mlir::ods::Builder {
public:
  // Re-export the ODS Parameter type so existing callers using
  // tblgen::Builder::Parameter continue to work.
  using Parameter = mlir::ods::Builder::Parameter;

  /// Constructs a Builder from the given Record instance.
  Builder(const llvm::Record *record, ArrayRef<SMLoc> loc);

  /// Returns the underlying TableGen record.
  const llvm::Record *getDef() const { return def; }

protected:
  const llvm::Record *def;
};

} // namespace tblgen
} // namespace mlir

#endif // MLIR_TABLEGEN_BUILDER_H_
