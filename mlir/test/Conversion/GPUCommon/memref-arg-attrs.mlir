// RUN: mlir-opt %s -split-input-file -convert-gpu-to-rocdl='use-bare-ptr-memref-call-conv=0' | FileCheck %s --check-prefixes=CHECK,ROCDL
// RUN: mlir-opt %s -split-input-file -convert-gpu-to-nvvm='use-bare-ptr-memref-call-conv=0' | FileCheck %s --check-prefixes=CHECK,NVVM

gpu.module @kernel {
  gpu.func @test_func_readonly(%arg0 : memref<f32> {llvm.readonly} ) {
    gpu.return
  }
}

// CHECK-LABEL:  llvm.func @test_func_readonly
// ROCDL-SAME:  !ptr.ptr {llvm.readonly}
//  NVVM-SAME:  !ptr.ptr {llvm.readonly}


// -----

gpu.module @kernel {
  gpu.func @test_func_writeonly(%arg0 : memref<f32> {llvm.writeonly} ) {
    gpu.return
  }
}

// CHECK-LABEL:  llvm.func @test_func_writeonly
// ROCDL-SAME:  !ptr.ptr {llvm.writeonly}
//  NVVM-SAME:  !ptr.ptr {llvm.writeonly}


// -----

gpu.module @kernel {
  gpu.func @test_func_nonnull(%arg0 : memref<f32> {llvm.nonnull} ) {
    gpu.return
  }
}

// CHECK-LABEL:  llvm.func @test_func_nonnull
// ROCDL-SAME:  !ptr.ptr {llvm.nonnull}
//  NVVM-SAME:  !ptr.ptr {llvm.nonnull}


// -----

gpu.module @kernel {
  gpu.func @test_func_dereferenceable(%arg0 : memref<f32> {llvm.dereferenceable = 4 : i64} ) {
    gpu.return
  }
}

// CHECK-LABEL:  llvm.func @test_func_dereferenceable
// ROCDL-SAME:  !ptr.ptr {llvm.dereferenceable = 4 : i64}
//  NVVM-SAME:  !ptr.ptr {llvm.dereferenceable = 4 : i64}


// -----

gpu.module @kernel {
  gpu.func @test_func_dereferenceable_or_null(%arg0 : memref<f32> {llvm.dereferenceable_or_null = 4 : i64} ) {
    gpu.return
  }
}

// CHECK-LABEL:  llvm.func @test_func_dereferenceable_or_null
// ROCDL-SAME:  !ptr.ptr {llvm.dereferenceable_or_null = 4 : i64}
//  NVVM-SAME:  !ptr.ptr {llvm.dereferenceable_or_null = 4 : i64}
