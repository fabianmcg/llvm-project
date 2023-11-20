// RUN: mlir-opt -convert-spirv-to-llvm -verify-diagnostics %s | FileCheck %s --check-prefixes=CHECK-UNKNOWN,CHECK-ALL
// RUN: mlir-opt -convert-spirv-to-llvm='client-api=OpenCL' -verify-diagnostics %s | FileCheck %s --check-prefixes=CHECK-OPENCL,CHECK-ALL

// CHECK-OPENCL:         llvm.func @pointerUniformConstant(!ptr.ptr<2>)
// CHECK-UNKNOWN:        llvm.func @pointerUniformConstant(!ptr.ptr)
spirv.func @pointerUniformConstant(!spirv.ptr<i1, UniformConstant>) "None"

// CHECK-OPENCL:         llvm.mlir.global external constant @varUniformConstant() {addr_space = 2 : i32} : i1
// CHECK-UNKNOWN:        llvm.mlir.global external constant @varUniformConstant() {addr_space = 0 : i32} : i1
spirv.GlobalVariable @varUniformConstant : !spirv.ptr<i1, UniformConstant>

// CHECK-OPENCL:         llvm.func @pointerInput(!ptr.ptr<1>)
// CHECK-UNKNOWN:        llvm.func @pointerInput(!ptr.ptr)
spirv.func @pointerInput(!spirv.ptr<i1, Input>) "None"

// CHECK-OPENCL:         llvm.mlir.global external constant @varInput() {addr_space = 1 : i32} : i1
// CHECK-UNKNOWN:        llvm.mlir.global external constant @varInput() {addr_space = 0 : i32} : i1
spirv.GlobalVariable @varInput : !spirv.ptr<i1, Input>

// CHECK-ALL:            llvm.func @pointerUniform(!ptr.ptr)
spirv.func @pointerUniform(!spirv.ptr<i1, Uniform>) "None"

// CHECK-ALL:            llvm.func @pointerOutput(!ptr.ptr)
spirv.func @pointerOutput(!spirv.ptr<i1, Output>) "None"

// CHECK-ALL:            llvm.mlir.global external @varOutput() {addr_space = 0 : i32} : i1
spirv.GlobalVariable @varOutput : !spirv.ptr<i1, Output>

// CHECK-OPENCL:         llvm.func @pointerWorkgroup(!ptr.ptr<3>)
// CHECK-UNKNOWN:        llvm.func @pointerWorkgroup(!ptr.ptr)
spirv.func @pointerWorkgroup(!spirv.ptr<i1, Workgroup>) "None"

// CHECK-OPENCL:         llvm.func @pointerCrossWorkgroup(!ptr.ptr<1>)
// CHECK-UNKNOWN:        llvm.func @pointerCrossWorkgroup(!ptr.ptr)
spirv.func @pointerCrossWorkgroup(!spirv.ptr<i1, CrossWorkgroup>) "None"

// CHECK-ALL:            llvm.func @pointerPrivate(!ptr.ptr)
spirv.func @pointerPrivate(!spirv.ptr<i1, Private>) "None"

// CHECK-ALL:            llvm.mlir.global private @varPrivate() {addr_space = 0 : i32} : i1
spirv.GlobalVariable @varPrivate : !spirv.ptr<i1, Private>

// CHECK-ALL:            llvm.func @pointerFunction(!ptr.ptr)
spirv.func @pointerFunction(!spirv.ptr<i1, Function>) "None"

// CHECK-OPENCL:         llvm.func @pointerGeneric(!ptr.ptr<4>)
// CHECK-UNKNOWN:         llvm.func @pointerGeneric(!ptr.ptr)
spirv.func @pointerGeneric(!spirv.ptr<i1, Generic>) "None"

// CHECK-ALL:            llvm.func @pointerPushConstant(!ptr.ptr)
spirv.func @pointerPushConstant(!spirv.ptr<i1, PushConstant>) "None"

// CHECK-ALL:            llvm.func @pointerAtomicCounter(!ptr.ptr)
spirv.func @pointerAtomicCounter(!spirv.ptr<i1, AtomicCounter>) "None"

// CHECK-ALL:            llvm.func @pointerImage(!ptr.ptr)
spirv.func @pointerImage(!spirv.ptr<i1, Image>) "None"

// CHECK-ALL:            llvm.func @pointerStorageBuffer(!ptr.ptr)
spirv.func @pointerStorageBuffer(!spirv.ptr<i1, StorageBuffer>) "None"

// CHECK-ALL:            llvm.mlir.global external @varStorageBuffer() {addr_space = 0 : i32} : i1
spirv.GlobalVariable @varStorageBuffer : !spirv.ptr<i1, StorageBuffer>

// CHECK-ALL:            llvm.func @pointerCallableDataKHR(!ptr.ptr)
spirv.func @pointerCallableDataKHR(!spirv.ptr<i1, CallableDataKHR>) "None"

// CHECK-ALL:            llvm.func @pointerIncomingCallableDataKHR(!ptr.ptr)
spirv.func @pointerIncomingCallableDataKHR(!spirv.ptr<i1, IncomingCallableDataKHR>) "None"

// CHECK-ALL:            llvm.func @pointerRayPayloadKHR(!ptr.ptr)
spirv.func @pointerRayPayloadKHR(!spirv.ptr<i1, RayPayloadKHR>) "None"

// CHECK-ALL:            llvm.func @pointerHitAttributeKHR(!ptr.ptr)
spirv.func @pointerHitAttributeKHR(!spirv.ptr<i1, HitAttributeKHR>) "None"

// CHECK-ALL:            llvm.func @pointerIncomingRayPayloadKHR(!ptr.ptr)
spirv.func @pointerIncomingRayPayloadKHR(!spirv.ptr<i1, IncomingRayPayloadKHR>) "None"

// CHECK-ALL:            llvm.func @pointerShaderRecordBufferKHR(!ptr.ptr)
spirv.func @pointerShaderRecordBufferKHR(!spirv.ptr<i1, ShaderRecordBufferKHR>) "None"

// CHECK-ALL:            llvm.func @pointerPhysicalStorageBuffer(!ptr.ptr)
spirv.func @pointerPhysicalStorageBuffer(!spirv.ptr<i1, PhysicalStorageBuffer>) "None"

// CHECK-ALL:            llvm.func @pointerCodeSectionINTEL(!ptr.ptr)
spirv.func @pointerCodeSectionINTEL(!spirv.ptr<i1, CodeSectionINTEL>) "None"

// CHECK-OPENCL:         llvm.func @pointerDeviceOnlyINTEL(!ptr.ptr<5>)
// CHECK-UNKNOWN:        llvm.func @pointerDeviceOnlyINTEL(!ptr.ptr)
spirv.func @pointerDeviceOnlyINTEL(!spirv.ptr<i1, DeviceOnlyINTEL>) "None"

// CHECK-OPENCL:         llvm.func @pointerHostOnlyINTEL(!ptr.ptr<6>)
// CHECK-UNKOWN:         llvm.func @pointerHostOnlyINTEL(!ptr.ptr)
spirv.func @pointerHostOnlyINTEL(!spirv.ptr<i1, HostOnlyINTEL>) "None"
