//===- Predicate.cpp - ODS predicate value type ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ods::Pred is a simple value type; all condition-computation logic lives in
// the tblgen::predFromRecord() free function in lib/TableGen/Predicate.cpp.
//
//===----------------------------------------------------------------------===//

#include "mlir/ODS/Predicate.h"

// ods::Pred is fully inline; nothing else to define here.
