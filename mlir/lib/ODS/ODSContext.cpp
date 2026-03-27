//===- ODSContext.cpp - ODS context and allocator -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/ODS/ODSContext.h"

using namespace mlir;
using namespace ods;

ODSContext::ODSContext() : strings(allocator) {}

ODSContext::~ODSContext() = default;

StringRef ODSContext::intern(StringRef str) {
  return strings.insert(str).first->getKey();
}
