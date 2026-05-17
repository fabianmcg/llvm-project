// RUN: mlir-opt %s -split-input-file | mlir-opt | FileCheck %s

// CHECK-LABEL: func @ptr_test
// CHECK: (%[[ARG0:.*]]: !ptr.ptr<#test.const_memory_space>, %[[ARG1:.*]]: !ptr.ptr<#test.const_memory_space<1>>)
// CHECK: -> (!ptr.ptr<#test.const_memory_space<1>>, !ptr.ptr<#test.const_memory_space>)
func.func @ptr_test(%arg0: !ptr.ptr<#test.const_memory_space>, %arg1: !ptr.ptr<#test.const_memory_space<1>>) -> (!ptr.ptr<#test.const_memory_space<1>>, !ptr.ptr<#test.const_memory_space>) {
  // CHECK: return %[[ARG1]], %[[ARG0]] : !ptr.ptr<#test.const_memory_space<1>>, !ptr.ptr<#test.const_memory_space>
  return %arg1, %arg0 : !ptr.ptr<#test.const_memory_space<1>>, !ptr.ptr<#test.const_memory_space>
}

// -----

// CHECK-LABEL: func @ptr_test
// CHECK: %[[ARG:.*]]: memref<!ptr.ptr<#test.const_memory_space>>
func.func @ptr_test(%arg0: memref<!ptr.ptr<#test.const_memory_space>>) {
  return
}

// CHECK-LABEL: func @ptr_test_1
// CHECK: (%[[ARG0:.*]]: !ptr.ptr<#test.const_memory_space>, %[[ARG1:.*]]: !ptr.ptr<#test.const_memory_space<3>>)
func.func @ptr_test_1(%arg0: !ptr.ptr<#test.const_memory_space>,
                      %arg1: !ptr.ptr<#test.const_memory_space<3>>) {
  return
}

// -----

// CHECK-LABEL: func @future_typed
// CHECK-SAME: !ptr.future<f32>
func.func @future_typed(%arg0: !ptr.future<f32>) {
  return
}

// -----

// CHECK-LABEL: func @future_empty
// CHECK-SAME: !ptr.future
func.func @future_empty(%arg0: !ptr.future) {
  return
}

// -----

// CHECK-LABEL: func @future_ptr_inner
// CHECK-SAME: !ptr.future<!ptr.ptr<#ptr.generic_space>>
func.func @future_ptr_inner(%arg0: !ptr.future<!ptr.ptr<#ptr.generic_space>>) {
  return
}
