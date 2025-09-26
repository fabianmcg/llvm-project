//===- InferStridedMetadataInterface.td - Strided MD Inference --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------===//
//
// Defines the interface for strided metadata analysis on buffer operations
//
//===-----------------------------------------------------===//

#ifndef MLIR_INTERFACES_INFERSTRIDEDMETADATAINTERFACE
#define MLIR_INTERFACES_INFERSTRIDEDMETADATAINTERFACE

include "mlir/IR/OpBase.td"

def InferStridedMetadataInterface : OpInterface<"InferStridedMetadataInterface"> {
  let description = [{
    Allows operations to participate in strided metadata analysis for buffer values by
    providing methods that allow them to specify bounds on offset, sizes, and strides
    of their result(s) given bounds on their input(s) if known.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
      Infer the strided metadata bounds on the results of this op given
      the bounds on its arguments.
    }],
    "void", "inferStridedMetadataRanges",
    (ins "::llvm::ArrayRef<::mlir::StridedMetadataRange>":$operands,
         "::mlir::GetIntRangeFn":$getIntRange,
         "::mlir::SetStridedMetadataRangeFn":$setMetadata,
         "int32_t":$indexBitwidth)>
  ];
}
#endif // MLIR_INTERFACES_INFERSTRIDEDMETADATAINTERFACE