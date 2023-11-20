// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s

module attributes {omp.is_target_device = true} {
  llvm.func @_QQmain() attributes {bindc_name = "main"} {
    %0 = llvm.mlir.addressof @_QFEsp : !ptr.ptr
    %1 = llvm.mlir.constant(10 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.mlir.constant(9 : index) : i64
    %5 = omp.bounds lower_bound(%3 : i64) upper_bound(%4 : i64) extent(%1 : i64) stride(%2 : i64) start_idx(%2 : i64)
    %6 = omp.map_info var_ptr(%0 : !ptr.ptr, !llvm.array<10 x i32>) map_clauses(tofrom) capture(ByRef) bounds(%5) -> !ptr.ptr {name = "sp"}
    omp.target map_entries(%6 -> %arg0 : !ptr.ptr) {
    ^bb0(%arg0: !ptr.ptr):
      %7 = llvm.mlir.constant(20 : i32) : i32
      %8 = llvm.mlir.constant(0 : i64) : i64
      %9 = llvm.getelementptr %arg0[0, %8] : (!ptr.ptr, i64) -> !ptr.ptr, !llvm.array<10 x i32>
      llvm.store %7, %9 : i32, !ptr.ptr
      %10 = llvm.mlir.constant(10 : i32) : i32
      %11 = llvm.mlir.constant(4 : i64) : i64
      %12 = llvm.getelementptr %arg0[0, %11] : (!ptr.ptr, i64) -> !ptr.ptr, !llvm.array<10 x i32>
      llvm.store %10, %12 : i32, !ptr.ptr
      omp.terminator
    }
    llvm.return
  }
  llvm.mlir.global internal @_QFEsp(dense<0> : tensor<10xi32>) {addr_space = 0 : i32} : !llvm.array<10 x i32>
  llvm.mlir.global external constant @_QQEnvironmentDefaults() {addr_space = 0 : i32} : !ptr.ptr {
    %0 = llvm.mlir.zero : !ptr.ptr
    llvm.return %0 : !ptr.ptr
  }
}


// CHECK: define {{.*}} void @__omp_offloading_{{.*}}_{{.*}}__QQmain_{{.*}}(ptr %{{.*}}, ptr %[[ARG1:.*]]) {

// CHECK: %[[ARG1_ALLOCA:.*]] = alloca ptr, align 8
// CHECK: store ptr %[[ARG1]], ptr %[[ARG1_ALLOCA]], align 8
// CHECK: %[[LOAD_ARG1_ALLOCA:.*]] = load ptr, ptr %[[ARG1_ALLOCA]], align 8
// CHECK: store i32 20, ptr %[[LOAD_ARG1_ALLOCA]], align 4
// CHECK: %[[GEP_ARG1_ALLOCA:.*]] = getelementptr inbounds [10 x i32], ptr %[[LOAD_ARG1_ALLOCA]], i32 0, i64 4
// CHECK: store i32 10, ptr %[[GEP_ARG1_ALLOCA]], align 4

