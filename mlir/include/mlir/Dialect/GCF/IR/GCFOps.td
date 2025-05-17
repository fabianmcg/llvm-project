//===- GCFOps.td - Generic Control Flow operations ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines MLIR generic control flow operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_GCF_GCFOPS
#define MLIR_DIALECT_GCF_GCFOPS

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

def GCF_Dialect : Dialect {
  let name = "gcf";
  let cppNamespace = "::mlir::gcf";

  let description = [{
    The `gcf` (generic control flow) dialect contains operations that
    represent control flow constructs such as `if` and `for`.
  }];

  let useDefaultAttributePrinterParser = 1;
}

// Base class for GCF dialect ops.
class GCF_Op<string mnemonic, list<Trait> traits = []> :
    Op<GCF_Dialect, mnemonic, traits>;

// Base class for GCF labels.
class GCF_LabelAttr<string name, string attr_mnemonic, list<Trait> traits = []>
    : CFGLabelAttrDef<GCF_Dialect, name, traits> {
  let cppClassName = name;
  let mnemonic = attr_mnemonic;
}

//===----------------------------------------------------------------------===//
// GCF labels
//===----------------------------------------------------------------------===//

def ContinueLabelAttr : GCF_LabelAttr<"ContinueLabel", "continue"> {
  let summary = "Continue label";
  let assemblyFormat = "";
}

def BreakLabelAttr : GCF_LabelAttr<"BreakLabel", "break"> {
  let summary = "Break label";
  let assemblyFormat = "";
}

//===----------------------------------------------------------------------===//
// BreakOp
//===----------------------------------------------------------------------===//

def BreakOp : GCF_Op<"break", [
  DeclareOpInterfaceMethods<CFGTerminatorOpInterface>,
  Pure,
  Terminator
]> {
  let summary = "Break op";
  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
  let assemblyFormat = [{
    attr-dict ($results^ `:` type($results))?
  }];
}

//===----------------------------------------------------------------------===//
// ContinueOp
//===----------------------------------------------------------------------===//

def ContinueOp : GCF_Op<"continue", [
  DeclareOpInterfaceMethods<CFGTerminatorOpInterface>,
  Pure,
  Terminator
]> {
  let summary = "Continue op";
  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
  let assemblyFormat = [{
    attr-dict ($results^ `:` type($results))?
  }];
}

//===----------------------------------------------------------------------===//
// ForOp
//===----------------------------------------------------------------------===//

def ForOp : GCF_Op<"for",
      [AutomaticAllocationScope, 
       AllTypesMatch<["lowerBound", "upperBound", "step"]>,
       ConditionallySpeculatable,
       DeclareOpInterfaceMethods<CFGOpInterface>,
       SingleBlock,
       RecursiveMemoryEffects]> {
  let summary = "for operation";
  let description = [{
  }];
  let arguments = (ins AnySignlessIntegerOrIndex:$lowerBound,
                       AnySignlessIntegerOrIndex:$upperBound,
                       AnySignlessIntegerOrIndex:$step,
                       Variadic<AnyType>:$initArgs);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$region);
  let extraClassDeclaration = [{
    using BodyBuilderFn =
        function_ref<void(OpBuilder &, Location, Value, ValueRange)>;

    Value getInductionVar() { return getBody()->getArgument(0); }

    /// Return the `index`-th region iteration argument.
    BlockArgument getRegionIterArg(unsigned index) {
      assert(index < getNumRegionIterArgs() &&
        "expected an index less than the number of region iter args");
      return getBody()->getArguments().drop_front(getNumInductionVars())[index];
    }

    void setLowerBound(Value bound) { getOperation()->setOperand(0, bound); }
    void setUpperBound(Value bound) { getOperation()->setOperand(1, bound); }
    void setStep(Value step) { getOperation()->setOperand(2, step); }

    /// Number of induction variables, always 1 for gcf::ForOp.
    unsigned getNumInductionVars() { return 1; }
    /// Number of region arguments for loop-carried values
    unsigned getNumRegionIterArgs() {
      return getBody()->getNumArguments() - getNumInductionVars();
    }
    /// Number of operands controlling the loop: lb, ub, step
    unsigned getNumControlOperands() { return 3; }

    /// Returns the step as an `APInt` if it is constant.
    std::optional<APInt> getConstantStep();

    /// Interface method for ConditionallySpeculatable.
    Speculation::Speculatability getSpeculatability();
    ::mlir::Block::BlockArgListType getRegionIterArgs();
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

def IfOp : GCF_Op<"if", [DeclareOpInterfaceMethods<CFGOpInterface>,
    InferTypeOpAdaptor, SingleBlock,
    RecursiveMemoryEffects, RecursivelySpeculatable, NoRegionArguments]> {
  let summary = "if-then-else operation";
  let description = [{
    The `gcf.if` operation represents an if-then-else construct for
    conditionally executing two regions of code. The operand to an if operation
    is a boolean value.
  }];
  let arguments = (ins I1:$condition);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$thenRegion,
                        MaxSizedRegion<1>:$elseRegion);

  let extraClassDeclaration = [{
    Block* thenBlock();
    YieldOp thenYield();
    Block* elseBlock();
    YieldOp elseYield();
  }];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

def YieldOp : GCF_Op<"yield", [Pure, DeclareOpInterfaceMethods<CFGTerminatorOpInterface>, Terminator]> {
  let summary = "loop yield and termination operation";
  let description = [{
    The `gcf.yield` operation yields an SSA value from the GCF dialect op region and
    terminates the regions. The semantics of how the values are yielded is
    defined by the parent operation.
    If `gcf.yield` has any operands, the operands must match the parent
    operation's results.
    If the parent operation defines no values, then the `gcf.yield` may be
    left out in the custom syntax and the builders will insert one implicitly.
    Otherwise, it has to be present in the syntax to indicate which values are
    yielded.
  }];

  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let assemblyFormat = [{
    attr-dict ($results^ `:` type($results))?
  }];
}

#endif // MLIR_DIALECT_GCF_GCFOPS
