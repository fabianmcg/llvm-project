//===-- PtrEnums.td - Ptr dialect enum file ----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef PTR_ENUMS
#define PTR_ENUMS

include "mlir/Dialect/Ptr/IR/PtrDialect.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Base classes for Ptr enum attributes.
//===----------------------------------------------------------------------===//

class Ptr_EnumAttrCase<string cppSym, string irSym, string llvmSym, int val> :
    I64EnumAttrCase<cppSym, val, irSym> {
  // The name of the equivalent enumerant in LLVM.
  string llvmEnumerant = llvmSym;
}

class Ptr_EnumAttr<string name, string llvmName, string description,
                    list<Ptr_EnumAttrCase> cases,
                    list<Ptr_EnumAttrCase> unsupportedCases = []> :
    I64EnumAttr<name, description, cases> {
  // List of unsupported cases that have no conversion to an MLIR value.
  list<Ptr_EnumAttrCase> unsupported = unsupportedCases;

  // The equivalent enum class name in LLVM.
  string llvmClassName = llvmName;
}

def AtomicOrderingNotAtomic : Ptr_EnumAttrCase<"not_atomic",
                                                "not_atomic", "NotAtomic", 0>;
def AtomicOrderingUnordered : Ptr_EnumAttrCase<"unordered",
                                                "unordered", "Unordered", 1>;
def AtomicOrderingMonotonic : Ptr_EnumAttrCase<"monotonic",
                                                "monotonic", "Monotonic", 2>;
def AtomicOrderingAcquire   : Ptr_EnumAttrCase<"acquire",
                                                "acquire", "Acquire", 4>;
def AtomicOrderingRelease   : Ptr_EnumAttrCase<"release",
                                                "release", "Release", 5>;
def AtomicOrderingAcquireRelease :
      Ptr_EnumAttrCase<"acq_rel", "acq_rel", "AcquireRelease", 6>;
def AtomicOrderingSequentiallyConsistent :
      Ptr_EnumAttrCase<"seq_cst", "seq_cst", "SequentiallyConsistent", 7>;

def AtomicOrdering : Ptr_EnumAttr<
    "AtomicOrdering",
    "::llvm::AtomicOrdering",
    "Atomic ordering for LLVM's memory model",
    [AtomicOrderingNotAtomic, AtomicOrderingUnordered, AtomicOrderingMonotonic,
     AtomicOrderingAcquire, AtomicOrderingRelease, AtomicOrderingAcquireRelease,
     AtomicOrderingSequentiallyConsistent
    ]> {
  let cppNamespace = "::mlir::ptr";
}

#endif // PTR_ENUMS
