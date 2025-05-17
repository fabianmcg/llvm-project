//===-- ControlFlowGraphInterfaces.td - ControlFlow Interfaces ----------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a set of interfaces that can be used to define information
// about control flow operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_CONTROLFLOWGRAPHINTERFACES
#define MLIR_IR_CONTROLFLOWGRAPHINTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// CFGOpInterface
//===----------------------------------------------------------------------===//

def CFGOpInterface : OpInterface<"CFGOpInterface"> {
  let description = [{
    This interface provides information for operations that accept or forward
    control-flow. CFG operations on entry can forward control-flow to any of
    the regions owned by the operation, to the operation itself or a symbol
    that is a control-flow scope.
    When the operation accepts control-flow, it can forward control-flow back
    to a region owned by the operation, to the next operation in the block or
    a symbol that is a control-flow scope.

    This interface is meant to model well-defined cases of control-flow and
    value propagation, where what occurs along control-flow edges is assumed to
    be side-effect free.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
        Returns the operands of this operation that are forwarded to the  successor's when branching to `point` on entry.
      }],
      "::mlir::OperandRange", "getOnEntrySuccessorOperands",
      (ins "::mlir::CFGBranchPoint":$point)
    >,
    InterfaceMethod<[{
        Returns the potential successors when executing the op on entry.

        Note: The control flow does not necessarily have to enter any region of
        this operation.
      }],
      "void", "getOnEntrySuccessors",
      (ins "std::optional<::llvm::ArrayRef<::mlir::Attribute>>":$operands,
           "::llvm::SmallVectorImpl<::mlir::CFGSuccessor> &":$successors)
    >,
    InterfaceMethod<[{
        Returns the potential successors when accepting control from `label`.
      }],
      "void", "getLabelSuccessors",
      (ins "::mlir::CFGLabel":$label,
           "::llvm::SmallVectorImpl<::mlir::CFGSuccessor> &":$successors)
    >,
    InterfaceMethod<[{
        This method returns the control-flow terminators accepted by this
        operation.
      }],
      "void", "getAcceptedTerminators",
      (ins "const ::llvm::DenseSet<std::pair<::mlir::CFGLabel, ::mlir::CFGTerminatorOpInterface>>&":$terminators,
           "::llvm::SmallVectorImpl<std::pair<::mlir::CFGLabel, ::mlir::CFGTerminatorOpInterface>> &":$acceptedTerminators)
    >,
  ];
}

//===----------------------------------------------------------------------===//
// CFGTerminatorOpInterface
//===----------------------------------------------------------------------===//

def CFGTerminatorOpInterface :
  OpInterface<"CFGTerminatorOpInterface"> {
  let description = [{
    This interface provides information for CFG branching terminator operations.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
        Returns a mutable range of operands that are semantically "returned" by
        passing them to the successor indicated by `label`.
      }],
      "::mlir::MutableOperandRange", "getLabelMutableSuccessorOperands",
      (ins "::mlir::CFGLabel":$label)
    >,
    InterfaceMethod<[{
        Returns the potential successor labels that are branched to after this
        terminator.

        If `operands` holds a value, this method also passes along the constant operands of this op.
        `operands` contains an entry for every operand of this op, with a null
        attribute if the operand has no constant value.
      }],
      "void", "getSuccessorLabels",
      (ins "std::optional<::llvm::ArrayRef<::mlir::Attribute>>":$operands,
           "::llvm::SmallVectorImpl<::mlir::CFGLabel>&":$labels)
    >,
  ];
  let extraClassDeclaration = [{
    // Returns a range of operands that are semantically "returned" by passing
    // them to the successor given by `label`.
    ::mlir::OperandRange getLabelSuccessorOperands(::mlir::CFGLabel label) {
      return getLabelMutableSuccessorOperands(label);
    }
  }];
}

#endif // MLIR_IR_CONTROLFLOWGRAPHINTERFACES
