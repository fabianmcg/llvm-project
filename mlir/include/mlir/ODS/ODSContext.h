//===- ODSContext.h - ODS context and allocator -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines ODSContext, which owns and allocates ODS model objects.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ODS_ODSCONTEXT_H_
#define MLIR_ODS_ODSCONTEXT_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Allocator.h"

namespace mlir {
namespace ods {

/// Owns and allocates ODS model objects. Callers must ensure the context
/// outlives all objects allocated from it. Supports future uniquing extensions.
class ODSContext {
public:
  ODSContext();
  ~ODSContext();

  // Disallow copy and assignment.
  ODSContext(const ODSContext &) = delete;
  ODSContext &operator=(const ODSContext &) = delete;

  /// Returns the bump-pointer allocator used to allocate ODS objects.
  llvm::BumpPtrAllocator &getAllocator() { return allocator; }

  /// Interns a string into the context and returns a stable StringRef.
  StringRef intern(StringRef str);

private:
  llvm::BumpPtrAllocator allocator;
  llvm::StringSet<llvm::BumpPtrAllocator &> strings;
};

} // namespace ods
} // namespace mlir

#endif // MLIR_ODS_ODSCONTEXT_H_
