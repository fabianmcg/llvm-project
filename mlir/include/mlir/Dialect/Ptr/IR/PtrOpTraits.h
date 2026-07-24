//===- PtrOpTraits.h - Pointer dialect op traits ----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares op traits for the Ptr dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTR_IR_PTROPTRAITS_H
#define MLIR_DIALECT_PTR_IR_PTROPTRAITS_H

#include "mlir/Dialect/Ptr/IR/PtrTypes.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {

namespace ptr::impl {
/// Verifies that if `futureValue` is a `FutureType`, its memory space matches
/// the memory space of `ptrType`. `futureValue` may be a null `Value`.
LogicalResult verifyFutureTrait(Operation *op, ptr::PtrType ptrType,
                                Value futureValue);
} // namespace ptr::impl

namespace OpTrait {
/// Op trait for operations that associate a pointer with a future result.
/// Requires the concrete op to expose:
/// - `ptr::PtrType getPtrType()` — the pointer type driving the operation.
/// - `Value getFuture()` — the future value (may be null or non-future typed).
/// Verifies that when the future value has a `ptr::FutureType`, its memory
/// space matches the one in the pointer type.
template <typename ConcreteOp>
class FutureVerifierOpTrait
    : public TraitBase<ConcreteOp, FutureVerifierOpTrait> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    auto concreteOp = cast<ConcreteOp>(op);
    return ptr::impl::verifyFutureTrait(op, concreteOp.getPtrType(),
                                        concreteOp.getFuture());
  }
};
} // namespace OpTrait

} // namespace mlir

#endif // MLIR_DIALECT_PTR_IR_PTROPTRAITS_H
