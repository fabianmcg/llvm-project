// RUN: mlir-cpu-runner %s -e caller --entry-point-result=i32 | FileCheck %s
// Varaidic argument list (va_list) and the extraction logics are ABI-specific.
// REQUIRES: x86-native-target
// UNSUPPORTED: system-windows

// Check if variadic functions can be called and the correct variadic argument
// can be extracted.

llvm.func @caller() -> i32 {
  %0 = llvm.mlir.constant(3 : i32) : i32
  %1 = llvm.mlir.constant(2 : i32) : i32
  %2 = llvm.mlir.constant(1 : i32) : i32
  %3 = llvm.call @foo(%2, %1, %0) vararg(!llvm.func<i32 (i32, ...)>) : (i32, i32, i32) -> i32
  llvm.return %3 : i32
}

// Equivalent C code:
// int foo(int X, ...) {
//  va_list args;
//  va_start(args, X);
//  int num = va_arg(args, int);
//  va_end(args);
//  return num;
//}
llvm.func @foo(%arg0: i32, ...) -> i32 {
  %0 = llvm.mlir.constant(8 : i64) : i64
  %1 = llvm.mlir.constant(2 : i32) : i32
  %2 = llvm.mlir.constant(0 : i64) : i64
  %3 = llvm.mlir.constant(0 : i64) : i64
  %4 = llvm.mlir.constant(8 : i32) : i32
  %5 = llvm.mlir.constant(3 : i32) : i32
  %6 = llvm.mlir.constant(0 : i64) : i64
  %7 = llvm.mlir.constant(0 : i64) : i64
  %8 = llvm.mlir.constant(41 : i32) : i32
  %9 = llvm.mlir.constant(0 : i32) : i32
  %10 = llvm.mlir.constant(0 : i64) : i64
  %11 = llvm.mlir.constant(0 : i64) : i64
  %12 = llvm.mlir.constant(1 : i32) : i32
  %13 = llvm.alloca %12 x !llvm.array<1 x struct<"struct.va_list", (i32, i32, ptr, ptr)>> {alignment = 8 : i64} : (i32) -> !ptr.ptr
  llvm.intr.vastart %13 : !ptr.ptr
  %15 = llvm.getelementptr %13[%11, %10, 0] : (!ptr.ptr, i64, i64) -> !ptr.ptr, !llvm.array<1 x struct<"struct.va_list", (i32, i32, ptr, ptr)>>
  %16 = llvm.load %15 : !ptr.ptr -> i32
  %17 = llvm.icmp "ult" %16, %8 : i32
  llvm.cond_br %17, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  %18 = llvm.getelementptr %13[%7, %6, 3] : (!ptr.ptr, i64, i64) -> !ptr.ptr, !llvm.array<1 x struct<"struct.va_list", (i32, i32, ptr, ptr)>>
  %19 = llvm.load %18 : !ptr.ptr -> !ptr.ptr
  %20 = llvm.zext %16 : i32 to i64
  %21 = llvm.getelementptr %19[%20] : (!ptr.ptr, i64) -> !ptr.ptr, i8
  %22 = llvm.add %16, %4  : i32
  llvm.store %22, %15 : i32, !ptr.ptr
  llvm.br ^bb3(%21 : !ptr.ptr)
^bb2:  // pred: ^bb0
  %23 = llvm.getelementptr %13[%3, %2, 2] : (!ptr.ptr, i64, i64) -> !ptr.ptr, !llvm.array<1 x struct<"struct.va_list", (i32, i32, ptr, ptr)>>
  %24 = llvm.load %23 : !ptr.ptr -> !ptr.ptr
  %25 = llvm.getelementptr %24[%0] : (!ptr.ptr, i64) -> !ptr.ptr, i8
  llvm.store %25, %23 : !ptr.ptr, !ptr.ptr
  llvm.br ^bb3(%24 : !ptr.ptr)
^bb3(%26: !ptr.ptr):  // 2 preds: ^bb1, ^bb2
  %28 = llvm.load %26 : !ptr.ptr -> i32
  llvm.intr.vaend %13 : !ptr.ptr
  llvm.return %28 : i32
}

// CHECK: 2
