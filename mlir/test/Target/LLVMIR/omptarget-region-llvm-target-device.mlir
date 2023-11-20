// This test checks that the name of the generated kernel function is using the
// name stored in the omp.outline_parent_name attribute.
// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

module attributes {omp.is_target_device = true} {
  llvm.func @writeindex_omp_outline_0_(%val0: !ptr.ptr, %val1: !ptr.ptr) attributes {omp.outline_parent_name = "writeindex_"} {
    %0 = omp.map_info var_ptr(%val0 : !ptr.ptr, i32)   map_clauses(tofrom) capture(ByRef) -> !ptr.ptr {name = ""}
    %1 = omp.map_info var_ptr(%val1 : !ptr.ptr, i32)   map_clauses(tofrom) capture(ByRef) -> !ptr.ptr {name = ""}
    omp.target   map_entries(%0 -> %arg0, %1 -> %arg1 : !ptr.ptr, !ptr.ptr) {
    ^bb0(%arg0: !ptr.ptr, %arg1: !ptr.ptr):
      %2 = llvm.mlir.constant(20 : i32) : i32
      %3 = llvm.mlir.constant(10 : i32) : i32
      llvm.store %3, %arg0 : i32, !ptr.ptr
      llvm.store %2, %arg1 : i32, !ptr.ptr
      omp.terminator
    }
    llvm.return
  }
}

// CHECK: define {{.*}} void @__omp_offloading_{{.*}}_{{.*}}_writeindex__l{{.*}}(ptr {{.*}}, ptr {{.*}}) {
