//===-- MemorySpaceInterfaces.td - Memory space interfaces ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines memory space attribute interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef PTR_MEMORYSPACEINTERFACES
#define PTR_MEMORYSPACEINTERFACES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Memory space attribute interface.
//===----------------------------------------------------------------------===//

def MemorySpaceAttrInterface : AttrInterface<"MemorySpaceAttrInterface"> {
  let description = [{
    This interface defines a common API for interacting with the memory model of
    a memory space and the operations in the pointer dialect, giving proper
    semantical meaning to the ops.

    Furthermore, this interface allows concepts such as read-only memory to be
    adequately modeled and enforced.
  }];
  let cppNamespace = "::mlir::ptr";
  let methods = [
    InterfaceMethod<[{
        Returns which dialect owns the memory model.
      }],
      "Dialect*", "getModelOwner", (ins),
      [{}],
      [{ return nullptr; }]
    >,
    InterfaceMethod<[{
        Returns the address space as an unsigned int.
      }],
      "unsigned", "getAddressSpace", (ins),
      [{}],
      [{ return 0; }]
    >,
    InterfaceMethod<[{
        Returns the address space as an unsigned int.
      }],
      "Attribute", "getDefaultMemorySpace", (ins)
    >,
    InterfaceMethod<[{
        Returns whether a type can be loaded from memory.
      }],
      "bool", "isValidLoad", (ins "Type":$type),
      [{}],
      [{ return true; }]
    >,
  ];
}

#endif // PTR_MEMORYSPACEINTERFACES
