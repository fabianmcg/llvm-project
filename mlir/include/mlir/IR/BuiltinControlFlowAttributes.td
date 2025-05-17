//===- BuiltinControlFlowAttributes.td - Builtin CFG attributes ---------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the set of builtin MLIR control-flow attributes.
//
//===----------------------------------------------------------------------===//

#ifndef BUILTIN_CONTROLFLOW_ATTRIBUTES_TD
#define BUILTIN_CONTROLFLOW_ATTRIBUTES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinDialect.td"

// Base class for Builtin dialect CFG attributes.
class Builtin_CFGAttr<string name, string mnemonicArg, list<Trait> traits = []>
    : CFGLabelAttrDef<Builtin_Dialect, name, traits> {
  let cppClassName = name;
  let mnemonic = mnemonicArg;
}

//===----------------------------------------------------------------------===//
// ReturnControl
//===----------------------------------------------------------------------===//

def Builtin_ReturnControlAttr :
    Builtin_CFGAttr<"ReturnControlAttr", "return"> {
  let summary = "A CFG attribute returning control to a function";
  let description = [{
    A `return` attribute represents an attribute returning control to the parent
    function.

    Examples:

    ```mlir
    builtin.return
    ```
  }];
  let assemblyFormat = "";
}

//===----------------------------------------------------------------------===//
// YieldToParent
//===----------------------------------------------------------------------===//

def Builtin_YieldToParentAttr :
    Builtin_CFGAttr<"YieldToParentAttr", "yield_to_parent"> {
  let summary = "A CFG attribute yielding to the parent operation";
  let description = [{
    A `yield_to_parent` attribute represents an attribute yielding to the parent
    operation.

    Examples:

    ```mlir
    builtin.yield_to_parent
    ```
  }];
  let assemblyFormat = "";
}

//===----------------------------------------------------------------------===//
// YieldToRegion
//===----------------------------------------------------------------------===//

def Builtin_YieldToRegionAttr :
    Builtin_CFGAttr<"YieldToRegionAttr", "yield_to_region"> {
  let summary = "A CFG attribute yielding to a region in the parent operation";
  let description = [{
    A `yield_to_region` attribute represents an attribute yielding to a region
    in the parent operation.

    Examples:

    ```mlir
    builtin.yield_to_region<0>
    ```
  }];
  let parameters = (ins "unsigned":$region_number);
  let builders = [
    AttrBuilderWithInferredContext<(ins "Region&":$region), [{
      return Base::get(region.getContext(), region.getRegionNumber());
    }]>,
  ];
  let assemblyFormat = "`<` $region_number `>`";
}

#endif // BUILTIN_CONTROLFLOW_ATTRIBUTES_TD
