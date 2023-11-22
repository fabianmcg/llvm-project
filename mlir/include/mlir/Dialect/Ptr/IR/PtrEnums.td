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

// Case of the LLVM enum attribute backed by I64Attr with customized string
// representation that corresponds to what is visible in the textual IR form.
// The parameters are as follows:
//   - `cppSym`: name of the C++ enumerant for this case in MLIR API;
//   - `irSym`: keyword used in the custom form of MLIR operation;
//   - `llvmSym`: name of the C++ enumerant for this case in LLVM API.
// For example, `Ptr_EnumAttrCase<"Weak", "weak", "WeakAnyLinkage">` is usable
// as `<MlirEnumName>::Weak` in MLIR API, `WeakAnyLinkage` in LLVM API and
// is printed/parsed as `weak` in MLIR custom textual format.
class Ptr_EnumAttrCase<string cppSym, string irSym, string llvmSym, int val> :
    I64EnumAttrCase<cppSym, val, irSym> {
  // The name of the equivalent enumerant in LLVM.
  string llvmEnumerant = llvmSym;
}

// LLVM enum attribute backed by I64Attr with string representation
// corresponding to what is visible in the textual IR form.
// The parameters are as follows:
//   - `name`: name of the C++ enum class in MLIR API;
//   - `llvmName`: name of the C++ enum in LLVM API;
//   - `description`: textual description for documentation purposes;
//   - `cases`: list of enum cases;
//   - `unsupportedCases`: optional list of unsupported enum cases.
// For example, `Ptr_EnumAttr<Linkage, "::llvm::GlobalValue::LinkageTypes`
// produces `mlir::ptr::Linkage` enum class in MLIR API that corresponds to (a
// subset of) values in the `llvm::GlobalValue::LinkageTypes` in LLVM API.
// All unsupported cases are excluded from the MLIR enum and trigger an error
// during the import from LLVM IR. They are useful to handle sentinel values
// such as `llvm::AtomicRMWInst::BinOp::BAD_BINOP` that LLVM commonly uses to
// terminate its enums.
class Ptr_EnumAttr<string name, string llvmName, string description,
                    list<Ptr_EnumAttrCase> cases,
                    list<Ptr_EnumAttrCase> unsupportedCases = []> :
    I64EnumAttr<name, description, cases> {
  // List of unsupported cases that have no conversion to an MLIR value.
  list<Ptr_EnumAttrCase> unsupported = unsupportedCases;

  // The equivalent enum class name in LLVM.
  string llvmClassName = llvmName;
}

//===----------------------------------------------------------------------===//
// Atomic binary op enum attribute
//===----------------------------------------------------------------------===//

def AtomicBinOpXchg : Ptr_EnumAttrCase<"xchg", "xchg", "Xchg", 0>;
def AtomicBinOpAdd  : Ptr_EnumAttrCase<"add", "add", "Add", 1>;
def AtomicBinOpSub  : Ptr_EnumAttrCase<"sub", "sub", "Sub", 2>;
def AtomicBinOpAnd  : Ptr_EnumAttrCase<"_and", "_and", "And", 3>;
def AtomicBinOpNand : Ptr_EnumAttrCase<"nand", "nand", "Nand", 4>;
def AtomicBinOpOr   : Ptr_EnumAttrCase<"_or", "_or", "Or", 5>;
def AtomicBinOpXor  : Ptr_EnumAttrCase<"_xor", "_xor", "Xor", 6>;
def AtomicBinOpMax  : Ptr_EnumAttrCase<"max", "max", "Max", 7>;
def AtomicBinOpMin  : Ptr_EnumAttrCase<"min", "min", "Min", 8>;
def AtomicBinOpUMax : Ptr_EnumAttrCase<"umax", "umax", "UMax", 9>;
def AtomicBinOpUMin : Ptr_EnumAttrCase<"umin", "umin", "UMin", 10>;
def AtomicBinOpFAdd : Ptr_EnumAttrCase<"fadd", "fadd", "FAdd", 11>;
def AtomicBinOpFSub : Ptr_EnumAttrCase<"fsub", "fsub", "FSub", 12>;
def AtomicBinOpFMax : Ptr_EnumAttrCase<"fmax", "fmax", "FMax", 13>;
def AtomicBinOpFMin : Ptr_EnumAttrCase<"fmin", "fmin", "FMin", 14>;
def AtomicBinOpUIncWrap : Ptr_EnumAttrCase<"uinc_wrap",
                                            "uinc_wrap", "UIncWrap", 15>;
def AtomicBinOpUDecWrap : Ptr_EnumAttrCase<"udec_wrap",
                                            "udec_wrap", "UDecWrap", 16>;

// A sentinel value that has no MLIR counterpart.
def AtomicBadBinOp : Ptr_EnumAttrCase<"", "", "BAD_BINOP", 0>;

def AtomicBinOp : Ptr_EnumAttr<
    "AtomicBinOp",
    "::llvm::AtomicRMWInst::BinOp",
    "llvm.atomicrmw binary operations",
    [AtomicBinOpXchg, AtomicBinOpAdd, AtomicBinOpSub, AtomicBinOpAnd,
     AtomicBinOpNand, AtomicBinOpOr, AtomicBinOpXor, AtomicBinOpMax,
     AtomicBinOpMin, AtomicBinOpUMax, AtomicBinOpUMin, AtomicBinOpFAdd,
     AtomicBinOpFSub, AtomicBinOpFMax, AtomicBinOpFMin, AtomicBinOpUIncWrap,
     AtomicBinOpUDecWrap],
    [AtomicBadBinOp]> {
  let cppNamespace = "::mlir::ptr";
}

//===----------------------------------------------------------------------===//
// Atomic ordering enum attribute
//===----------------------------------------------------------------------===//

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
