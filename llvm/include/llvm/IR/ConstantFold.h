//==-- ConstantFold.h - DL-independent Constant Folding Interface -*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the DataLayout-independent constant folding interface.
// When possible, the DataLayout-aware constant folding interface in
// Analysis/ConstantFolding.h should be preferred.
//
// These interfaces are used by the ConstantExpr::get* methods to automatically
// fold constants when possible.
//
// These operators may return a null object if they don't know how to perform
// the specified operation on the specified constant types.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_CONSTANTFOLD_H
#define LLVM_IR_CONSTANTFOLD_H

#include "llvm/IR/InstrTypes.h"
#include "llvm/Support/Compiler.h"
#include <optional>

namespace llvm {
  template <typename T> class ArrayRef;
  class Value;
  class Constant;
  class Type;

  // Constant fold various types of instruction...
  LLVM_ABI Constant *
  ConstantFoldCastInstruction(unsigned opcode, ///< The opcode of the cast
                              Constant *V,     ///< The source constant
                              Type *DestTy     ///< The destination type
  );
  LLVM_ABI Constant *ConstantFoldSelectInstruction(Constant *Cond, Constant *V1,
                                                   Constant *V2);
  LLVM_ABI Constant *ConstantFoldExtractElementInstruction(Constant *Val,
                                                           Constant *Idx);
  LLVM_ABI Constant *ConstantFoldInsertElementInstruction(Constant *Val,
                                                          Constant *Elt,
                                                          Constant *Idx);
  LLVM_ABI Constant *ConstantFoldShuffleVectorInstruction(Constant *V1,
                                                          Constant *V2,
                                                          ArrayRef<int> Mask);
  LLVM_ABI Constant *
  ConstantFoldExtractValueInstruction(Constant *Agg, ArrayRef<unsigned> Idxs);
  LLVM_ABI Constant *
  ConstantFoldInsertValueInstruction(Constant *Agg, Constant *Val,
                                     ArrayRef<unsigned> Idxs);
  LLVM_ABI Constant *ConstantFoldUnaryInstruction(unsigned Opcode, Constant *V);
  LLVM_ABI Constant *ConstantFoldBinaryInstruction(unsigned Opcode,
                                                   Constant *V1, Constant *V2);
  LLVM_ABI Constant *
  ConstantFoldCompareInstruction(CmpInst::Predicate Predicate, Constant *C1,
                                 Constant *C2);
  LLVM_ABI Constant *
  ConstantFoldGetElementPtr(Type *Ty, Constant *C,
                            std::optional<ConstantRange> InRange,
                            ArrayRef<Value *> Idxs);
} // End llvm namespace

#endif
