//===- Region.cpp - Region class ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Region wrapper to simplify using TableGen Record defining a MLIR Region.
// isVariadic() is now inlined via the cached ods::Constraint::variadic field.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Region.h"
