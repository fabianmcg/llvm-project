// RUN: mlir-opt -convert-openmp-to-llvm -split-input-file %s | FileCheck %s

// CHECK-LABEL: llvm.func @foo(i64, i64)
func.func private @foo(index, index)

// CHECK-LABEL: llvm.func @critical_block_arg
func.func @critical_block_arg() {
  // CHECK: omp.critical
  omp.critical {
  // CHECK-NEXT: ^[[BB0:.*]](%[[ARG1:.*]]: i64, %[[ARG2:.*]]: i64):
  ^bb0(%arg1: index, %arg2: index):
    // CHECK-NEXT: llvm.call @foo(%[[ARG1]], %[[ARG2]]) : (i64, i64) -> ()
    func.call @foo(%arg1, %arg2) : (index, index) -> ()
    omp.terminator
  }
  return
}

// -----

// CHECK-LABEL: llvm.func @master_block_arg
func.func @master_block_arg() {
  // CHECK: omp.master
  omp.master {
  // CHECK-NEXT: ^[[BB0:.*]](%[[ARG1:.*]]: i64, %[[ARG2:.*]]: i64):
  ^bb0(%arg1: index, %arg2: index):
    // CHECK-DAG: %[[CAST_ARG1:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : i64 to index
    // CHECK-DAG: %[[CAST_ARG2:.*]] = builtin.unrealized_conversion_cast %[[ARG2]] : i64 to index
    // CHECK-NEXT: "test.payload"(%[[CAST_ARG1]], %[[CAST_ARG2]]) : (index, index) -> ()
    "test.payload"(%arg1, %arg2) : (index, index) -> ()
    omp.terminator
  }
  return
}

// -----

// CHECK-LABEL: llvm.func @branch_loop
func.func @branch_loop() {
  %start = arith.constant 0 : index
  %end = arith.constant 0 : index
  // CHECK: omp.parallel
  omp.parallel {
    // CHECK-NEXT: llvm.br ^[[BB1:.*]](%{{[0-9]+}}, %{{[0-9]+}} : i64, i64
    cf.br ^bb1(%start, %end : index, index)
  // CHECK-NEXT: ^[[BB1]](%[[ARG1:[0-9]+]]: i64, %[[ARG2:[0-9]+]]: i64):{{.*}}
  ^bb1(%0: index, %1: index):
    // CHECK-NEXT: %[[CMP:[0-9]+]] = llvm.icmp "slt" %[[ARG1]], %[[ARG2]] : i64
    %2 = arith.cmpi slt, %0, %1 : index
    // CHECK-NEXT: llvm.cond_br %[[CMP]], ^[[BB2:.*]](%{{[0-9]+}}, %{{[0-9]+}} : i64, i64), ^[[BB3:.*]]
    cf.cond_br %2, ^bb2(%end, %end : index, index), ^bb3
  // CHECK-NEXT: ^[[BB2]](%[[ARG3:[0-9]+]]: i64, %[[ARG4:[0-9]+]]: i64):
  ^bb2(%3: index, %4: index):
    // CHECK-NEXT: llvm.br ^[[BB1]](%[[ARG3]], %[[ARG4]] : i64, i64)
    cf.br ^bb1(%3, %4 : index, index)
  // CHECK-NEXT: ^[[BB3]]:
  ^bb3:
    omp.flush
    omp.barrier
    omp.taskwait
    omp.taskyield
    omp.terminator
  }
  return
}

// -----

// CHECK-LABEL: @wsloop
// CHECK: (%[[ARG0:.*]]: i64, %[[ARG1:.*]]: i64, %[[ARG2:.*]]: i64, %[[ARG3:.*]]: i64, %[[ARG4:.*]]: i64, %[[ARG5:.*]]: i64)
func.func @wsloop(%arg0: index, %arg1: index, %arg2: index, %arg3: index, %arg4: index, %arg5: index) {
  // CHECK: omp.parallel
  omp.parallel {
    // CHECK: omp.wsloop for (%[[ARG6:.*]], %[[ARG7:.*]]) : i64 = (%[[ARG0]], %[[ARG1]]) to (%[[ARG2]], %[[ARG3]]) step (%[[ARG4]], %[[ARG5]]) {
    "omp.wsloop"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) ({
    ^bb0(%arg6: index, %arg7: index):
      // CHECK-DAG: %[[CAST_ARG6:.*]] = builtin.unrealized_conversion_cast %[[ARG6]] : i64 to index
      // CHECK-DAG: %[[CAST_ARG7:.*]] = builtin.unrealized_conversion_cast %[[ARG7]] : i64 to index
      // CHECK: "test.payload"(%[[CAST_ARG6]], %[[CAST_ARG7]]) : (index, index) -> ()
      "test.payload"(%arg6, %arg7) : (index, index) -> ()
      omp.yield
    }) {operandSegmentSizes = array<i32: 2, 2, 2, 0, 0, 0, 0>} : (index, index, index, index, index, index) -> ()
    omp.terminator
  }
  return
}

// -----

// CHECK-LABEL: @atomic_write
// CHECK: (%[[ARG0:.*]]: !ptr.ptr)
// CHECK: %[[VAL0:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: omp.atomic.write %[[ARG0]] = %[[VAL0]] memory_order(relaxed) : !ptr.ptr, i32
func.func @atomic_write(%a: !ptr.ptr) -> () {
  %1 = arith.constant 1 : i32
  omp.atomic.write %a = %1 hint(none) memory_order(relaxed) : !ptr.ptr, i32
  return
}

// -----

// CHECK-LABEL: @atomic_read
// CHECK: (%[[ARG0:.*]]: !ptr.ptr, %[[ARG1:.*]]: !ptr.ptr)
// CHECK: omp.atomic.read %[[ARG1]] = %[[ARG0]] memory_order(acquire) hint(contended) : !ptr.ptr
func.func @atomic_read(%a: !ptr.ptr, %b: !ptr.ptr) -> () {
  omp.atomic.read %b = %a memory_order(acquire) hint(contended) : !ptr.ptr, i32
  return
}

// -----

func.func @atomic_update() {
  %0 = llvm.mlir.addressof @_QFsEc : !ptr.ptr
  omp.atomic.update   %0 : !ptr.ptr {
  ^bb0(%arg0: i32):
    %1 = arith.constant 1 : i32
    %2 = arith.addi %arg0, %1  : i32
    omp.yield(%2 : i32)
  }
  return
}
llvm.mlir.global internal @_QFsEc() : i32 {
  %0 = arith.constant 10 : i32
  llvm.return %0 : i32
}

// CHECK-LABEL: @atomic_update
// CHECK: %[[GLOBAL_VAR:.*]] = llvm.mlir.addressof @_QFsEc : !ptr.ptr
// CHECK: omp.atomic.update   %[[GLOBAL_VAR]] : !ptr.ptr {
// CHECK: ^bb0(%[[IN_VAL:.*]]: i32):
// CHECK:   %[[CONST_1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:   %[[OUT_VAL:.*]] = llvm.add %[[IN_VAL]], %[[CONST_1]]  : i32
// CHECK:   omp.yield(%[[OUT_VAL]] : i32)
// CHECK: }

// -----

// CHECK-LABEL: @threadprivate
// CHECK: (%[[ARG0:.*]]: !ptr.ptr)
// CHECK: %[[VAL0:.*]] = omp.threadprivate %[[ARG0]] : !ptr.ptr -> !ptr.ptr
func.func @threadprivate(%a: !ptr.ptr) -> () {
  %1 = omp.threadprivate %a : !ptr.ptr -> !ptr.ptr
  return
}

// -----

// CHECK:      llvm.func @simdloop_block_arg(%[[LOWER:.*]]: i32, %[[UPPER:.*]]: i32, %[[ITER:.*]]: i64) {
// CHECK:      omp.simdloop   for  (%[[ARG_0:.*]]) : i32 =
// CHECK-SAME:     (%[[LOWER]]) to (%[[UPPER]]) inclusive step (%[[LOWER]]) {
// CHECK:      llvm.br ^[[BB1:.*]](%[[ITER]] : i64)
// CHECK:        ^[[BB1]](%[[VAL_0:.*]]: i64):
// CHECK:          %[[VAL_1:.*]] = llvm.icmp "slt" %[[VAL_0]], %[[ITER]] : i64
// CHECK:          llvm.cond_br %[[VAL_1]], ^[[BB2:.*]], ^[[BB3:.*]]
// CHECK:        ^[[BB2]]:
// CHECK:          %[[VAL_2:.*]] = llvm.add %[[VAL_0]], %[[ITER]]  : i64
// CHECK:          llvm.br ^[[BB1]](%[[VAL_2]] : i64)
// CHECK:        ^[[BB3]]:
// CHECK:          omp.yield
func.func @simdloop_block_arg(%val : i32, %ub : i32, %i : index) {
  omp.simdloop   for  (%arg0) : i32 = (%val) to (%ub) inclusive step (%val) {
    cf.br ^bb1(%i : index)
  ^bb1(%0: index):
    %1 = arith.cmpi slt, %0, %i : index
    cf.cond_br %1, ^bb2, ^bb3
  ^bb2:
    %2 = arith.addi %0, %i : index
    cf.br ^bb1(%2 : index)
  ^bb3:
    omp.yield
  }
  return
}

// -----

// CHECK-LABEL: @task_depend
// CHECK:  (%[[ARG0:.*]]: !ptr.ptr) {
// CHECK:  omp.task depend(taskdependin -> %[[ARG0]] : !ptr.ptr) {
// CHECK:    omp.terminator
// CHECK:  }
// CHECK:   llvm.return
// CHECK: }

func.func @task_depend(%arg0: !ptr.ptr) {
  omp.task depend(taskdependin -> %arg0 : !ptr.ptr) {
    omp.terminator
  }
  return
}

// -----

// CHECK-LABEL: @_QPomp_target_data
// CHECK: (%[[ARG0:.*]]: !ptr.ptr, %[[ARG1:.*]]: !ptr.ptr, %[[ARG2:.*]]: !ptr.ptr, %[[ARG3:.*]]: !ptr.ptr)
// CHECK: %[[MAP0:.*]] = omp.map_info var_ptr(%[[ARG0]] : !ptr.ptr, i32)   map_clauses(to) capture(ByRef) -> !ptr.ptr {name = ""}
// CHECK: %[[MAP1:.*]] = omp.map_info var_ptr(%[[ARG1]] : !ptr.ptr, i32)   map_clauses(to) capture(ByRef) -> !ptr.ptr {name = ""}
// CHECK: %[[MAP2:.*]] = omp.map_info var_ptr(%[[ARG2]] : !ptr.ptr, i32)   map_clauses(always, exit_release_or_enter_alloc) capture(ByRef) -> !ptr.ptr {name = ""}
// CHECK: omp.target_enter_data map_entries(%[[MAP0]], %[[MAP1]], %[[MAP2]] : !ptr.ptr, !ptr.ptr, !ptr.ptr)
// CHECK: %[[MAP3:.*]] = omp.map_info var_ptr(%[[ARG0]] : !ptr.ptr, i32)   map_clauses(from) capture(ByRef) -> !ptr.ptr {name = ""}
// CHECK: %[[MAP4:.*]] = omp.map_info var_ptr(%[[ARG1]] : !ptr.ptr, i32)   map_clauses(from) capture(ByRef) -> !ptr.ptr {name = ""}
// CHECK: %[[MAP5:.*]] = omp.map_info var_ptr(%[[ARG2]] : !ptr.ptr, i32)   map_clauses(exit_release_or_enter_alloc) capture(ByRef) -> !ptr.ptr {name = ""}
// CHECK: %[[MAP6:.*]] = omp.map_info var_ptr(%[[ARG3]] : !ptr.ptr, i32)   map_clauses(always, delete) capture(ByRef) -> !ptr.ptr {name = ""}
// CHECK: omp.target_exit_data map_entries(%[[MAP3]], %[[MAP4]], %[[MAP5]], %[[MAP6]] : !ptr.ptr, !ptr.ptr, !ptr.ptr, !ptr.ptr)

llvm.func @_QPomp_target_data(%a : !ptr.ptr, %b : !ptr.ptr, %c : !ptr.ptr, %d : !ptr.ptr) {
  %0 = omp.map_info var_ptr(%a : !ptr.ptr, i32)   map_clauses(to) capture(ByRef) -> !ptr.ptr {name = ""}
  %1 = omp.map_info var_ptr(%b : !ptr.ptr, i32)   map_clauses(to) capture(ByRef) -> !ptr.ptr {name = ""}
  %2 = omp.map_info var_ptr(%c : !ptr.ptr, i32)   map_clauses(always, exit_release_or_enter_alloc) capture(ByRef) -> !ptr.ptr {name = ""}
  omp.target_enter_data map_entries(%0, %1, %2 : !ptr.ptr, !ptr.ptr, !ptr.ptr) {}
  %3 = omp.map_info var_ptr(%a : !ptr.ptr, i32)   map_clauses(from) capture(ByRef) -> !ptr.ptr {name = ""}
  %4 = omp.map_info var_ptr(%b : !ptr.ptr, i32)   map_clauses(from) capture(ByRef) -> !ptr.ptr {name = ""}
  %5 = omp.map_info var_ptr(%c : !ptr.ptr, i32)   map_clauses(exit_release_or_enter_alloc) capture(ByRef) -> !ptr.ptr {name = ""}
  %6 = omp.map_info var_ptr(%d : !ptr.ptr, i32)   map_clauses(always, delete) capture(ByRef) -> !ptr.ptr {name = ""}
  omp.target_exit_data map_entries(%3, %4, %5, %6 : !ptr.ptr, !ptr.ptr, !ptr.ptr, !ptr.ptr) {}
  llvm.return
}

// -----

// CHECK-LABEL: @_QPomp_target_data_region
// CHECK: (%[[ARG0:.*]]: !ptr.ptr, %[[ARG1:.*]]: !ptr.ptr) {
// CHECK: %[[MAP_0:.*]] = omp.map_info var_ptr(%[[ARG0]] : !ptr.ptr, !llvm.array<1024 x i32>)  map_clauses(tofrom) capture(ByRef) -> !ptr.ptr {name = ""}
// CHECK: omp.target_data map_entries(%[[MAP_0]] : !ptr.ptr) {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(10 : i32) : i32
// CHECK:           llvm.store %[[VAL_1]], %[[ARG1]] : i32, !ptr.ptr
// CHECK:           omp.terminator
// CHECK:         }
// CHECK:         llvm.return

llvm.func @_QPomp_target_data_region(%a : !ptr.ptr, %i : !ptr.ptr) {
  %1 = omp.map_info var_ptr(%a : !ptr.ptr, !llvm.array<1024 x i32>)   map_clauses(tofrom) capture(ByRef) -> !ptr.ptr {name = ""}
  omp.target_data map_entries(%1 : !ptr.ptr) {
    %2 = llvm.mlir.constant(10 : i32) : i32
    llvm.store %2, %i : i32, !ptr.ptr
    omp.terminator
  }
  llvm.return
}

// -----

// CHECK-LABEL:   llvm.func @_QPomp_target(
// CHECK:                             %[[ARG_0:.*]]: !ptr.ptr,
// CHECK:                             %[[ARG_1:.*]]: !ptr.ptr) {
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(64 : i32) : i32
// CHECK:           %[[MAP1:.*]] = omp.map_info var_ptr(%[[ARG_0]] : !ptr.ptr, !llvm.array<1024 x i32>)   map_clauses(tofrom) capture(ByRef) -> !ptr.ptr {name = ""}
// CHECK:           %[[MAP2:.*]] = omp.map_info var_ptr(%[[ARG_1]] : !ptr.ptr, i32)   map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !ptr.ptr {name = ""}
// CHECK:           omp.target   thread_limit(%[[VAL_0]] : i32) map_entries(%[[MAP1]] -> %[[BB_ARG0:.*]], %[[MAP2]] -> %[[BB_ARG1:.*]] : !ptr.ptr, !ptr.ptr) {
// CHECK:           ^bb0(%[[BB_ARG0]]: !ptr.ptr, %[[BB_ARG1]]: !ptr.ptr):
// CHECK:             %[[VAL_1:.*]] = llvm.mlir.constant(10 : i32) : i32
// CHECK:             llvm.store %[[VAL_1]], %[[BB_ARG1]] : i32, !ptr.ptr
// CHECK:             omp.terminator
// CHECK:           }
// CHECK:           llvm.return
// CHECK:         }

llvm.func @_QPomp_target(%a : !ptr.ptr, %i : !ptr.ptr) {
  %0 = llvm.mlir.constant(64 : i32) : i32
  %1 = omp.map_info var_ptr(%a : !ptr.ptr, !llvm.array<1024 x i32>)   map_clauses(tofrom) capture(ByRef) -> !ptr.ptr {name = ""}
  %3 = omp.map_info var_ptr(%i : !ptr.ptr, i32)   map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !ptr.ptr {name = ""}
  omp.target   thread_limit(%0 : i32) map_entries(%1 -> %arg0, %3 -> %arg1 : !ptr.ptr, !ptr.ptr) {
    ^bb0(%arg0: !ptr.ptr, %arg1: !ptr.ptr):
    %2 = llvm.mlir.constant(10 : i32) : i32
    llvm.store %2, %arg1 : i32, !ptr.ptr
    omp.terminator
  }
  llvm.return
}

// -----

// CHECK-LABEL: @_QPsb
// CHECK: omp.sections
// CHECK: omp.section
// CHECK: llvm.br
// CHECK: llvm.icmp
// CHECK: llvm.cond_br
// CHECK: llvm.br
// CHECK: omp.terminator
// CHECK: omp.terminator
// CHECK: llvm.return

llvm.func @_QPsb() {
  %0 = llvm.mlir.constant(0 : i64) : i64
  %1 = llvm.mlir.constant(10 : i64) : i64
  %2 = llvm.mlir.constant(1 : i64) : i64
  omp.sections   {
    omp.section {
      llvm.br ^bb1(%1 : i64)
    ^bb1(%3: i64):  // 2 preds: ^bb0, ^bb2
      %4 = llvm.icmp "sgt" %3, %0 : i64
      llvm.cond_br %4, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %5 = llvm.sub %3, %2  : i64
      llvm.br ^bb1(%5 : i64)
    ^bb3:  // pred: ^bb1
      omp.terminator
    }
    omp.terminator
  }
  llvm.return
}

// -----

// CHECK:  omp.reduction.declare @eqv_reduction : i32 init
// CHECK:  ^bb0(%{{.*}}: i32):
// CHECK:    %[[TRUE:.*]] = llvm.mlir.constant(true) : i1
// CHECK:    %[[TRUE_EXT:.*]] = llvm.zext %[[TRUE]] : i1 to i32
// CHECK:    omp.yield(%[[TRUE_EXT]] : i32)
// CHECK:  } combiner {
// CHECK:  ^bb0(%[[ARG_1:.*]]: i32, %[[ARG_2:.*]]: i32):
// CHECK:    %[[ZERO:.*]] = llvm.mlir.constant(0 : i64) : i32
// CHECK:    %[[CMP_1:.*]] = llvm.icmp "ne" %[[ARG_1]], %[[ZERO]] : i32
// CHECK:    %[[CMP_2:.*]] = llvm.icmp "ne" %[[ARG_2]], %[[ZERO]] : i32
// CHECK:    %[[COMBINE_VAL:.*]] = llvm.icmp "eq" %[[CMP_1]], %[[CMP_2]] : i1
// CHECK:    %[[COMBINE_VAL_EXT:.*]] = llvm.zext %[[COMBINE_VAL]] : i1 to i32
// CHECK:    omp.yield(%[[COMBINE_VAL_EXT]] : i32)
// CHECK-LABEL:  @_QPsimple_reduction
// CHECK:    %[[RED_ACCUMULATOR:.*]] = llvm.alloca %{{.*}} x i32 {bindc_name = "x", uniq_name = "_QFsimple_reductionEx"} : (i64) -> !ptr.ptr
// CHECK:    omp.parallel
// CHECK:      omp.wsloop reduction(@eqv_reduction -> %[[RED_ACCUMULATOR]] : !ptr.ptr) for
// CHECK:        omp.reduction %{{.*}}, %[[RED_ACCUMULATOR]] : i32, !ptr.ptr
// CHECK:        omp.yield
// CHECK:      omp.terminator
// CHECK:    llvm.return

omp.reduction.declare @eqv_reduction : i32 init {
^bb0(%arg0: i32):
  %0 = llvm.mlir.constant(true) : i1
  %1 = llvm.zext %0 : i1 to i32
  omp.yield(%1 : i32)
} combiner {
^bb0(%arg0: i32, %arg1: i32):
  %0 = llvm.mlir.constant(0 : i64) : i32
  %1 = llvm.icmp "ne" %arg0, %0 : i32
  %2 = llvm.icmp "ne" %arg1, %0 : i32
  %3 = llvm.icmp "eq" %1, %2 : i1
  %4 = llvm.zext %3 : i1 to i32
  omp.yield(%4 : i32)
}
llvm.func @_QPsimple_reduction(%arg0: !ptr.ptr {fir.bindc_name = "y"}) {
  %0 = llvm.mlir.constant(100 : i32) : i32
  %1 = llvm.mlir.constant(1 : i32) : i32
  %2 = llvm.mlir.constant(true) : i1
  %3 = llvm.mlir.constant(1 : i64) : i64
  %4 = llvm.alloca %3 x i32 {bindc_name = "x", uniq_name = "_QFsimple_reductionEx"} : (i64) -> !ptr.ptr
  %5 = llvm.zext %2 : i1 to i32
  llvm.store %5, %4 : i32, !ptr.ptr
  omp.parallel   {
    %6 = llvm.alloca %3 x i32 {adapt.valuebyref, in_type = i32, operandSegmentSizes = array<i32: 0, 0>, pinned} : (i64) -> !ptr.ptr
    omp.wsloop   reduction(@eqv_reduction -> %4 : !ptr.ptr) for  (%arg1) : i32 = (%1) to (%0) inclusive step (%1) {
      llvm.store %arg1, %6 : i32, !ptr.ptr
      %7 = llvm.load %6 : !ptr.ptr -> i32
      %8 = llvm.sext %7 : i32 to i64
      %9 = llvm.sub %8, %3  : i64
      %10 = llvm.getelementptr %arg0[0, %9] : (!ptr.ptr, i64) -> !ptr.ptr, !llvm.array<100 x i32>
      %11 = llvm.load %10 : !ptr.ptr -> i32
      omp.reduction %11, %4 : i32, !ptr.ptr
      omp.yield
    }
    omp.terminator
  }
  llvm.return
}

// -----

// CHECK-LABEL:  @_QQmain
llvm.func @_QQmain() {
  %0 = llvm.mlir.constant(0 : index) : i64
  %1 = llvm.mlir.constant(5 : index) : i64
  %2 = llvm.mlir.constant(1 : index) : i64
  %3 = llvm.mlir.constant(1 : i64) : i64
  %4 = llvm.alloca %3 x i32 : (i64) -> !ptr.ptr
// CHECK: omp.taskgroup
  omp.taskgroup   {
    %5 = llvm.trunc %2 : i64 to i32
    llvm.br ^bb1(%5, %1 : i32, i64)
  ^bb1(%6: i32, %7: i64):  // 2 preds: ^bb0, ^bb2
    %8 = llvm.icmp "sgt" %7, %0 : i64
    llvm.cond_br %8, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    llvm.store %6, %4 : i32, !ptr.ptr
// CHECK: omp.task
    omp.task   {
// CHECK: llvm.call @[[CALL_FUNC:.*]]({{.*}}) :
      llvm.call @_QFPdo_work(%4) : (!ptr.ptr) -> ()
// CHECK: omp.terminator
      omp.terminator
    }
    %9 = llvm.load %4 : !ptr.ptr -> i32
    %10 = llvm.add %9, %5  : i32
    %11 = llvm.sub %7, %2  : i64
    llvm.br ^bb1(%10, %11 : i32, i64)
  ^bb3:  // pred: ^bb1
    llvm.store %6, %4 : i32, !ptr.ptr
// CHECK: omp.terminator
    omp.terminator
  }
  llvm.return
}
// CHECK: @[[CALL_FUNC]]
llvm.func @_QFPdo_work(%arg0: !ptr.ptr {fir.bindc_name = "i"}) {
  llvm.return
}

// -----

// CHECK-LABEL:  @sub_
llvm.func @sub_() {
  %0 = llvm.mlir.constant(0 : index) : i64
  %1 = llvm.mlir.constant(1 : index) : i64
  %2 = llvm.mlir.constant(1 : i64) : i64
  %3 = llvm.alloca %2 x i32 {bindc_name = "i", in_type = i32, operandSegmentSizes = array<i32: 0, 0>, uniq_name = "_QFsubEi"} : (i64) -> !ptr.ptr
// CHECK: omp.ordered_region
  omp.ordered_region {
    %4 = llvm.trunc %1 : i64 to i32
    llvm.br ^bb1(%4, %1 : i32, i64)
  ^bb1(%5: i32, %6: i64):  // 2 preds: ^bb0, ^bb2
    %7 = llvm.icmp "sgt" %6, %0 : i64
    llvm.cond_br %7, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    llvm.store %5, %3 : i32, !ptr.ptr
    %8 = llvm.load %3 : !ptr.ptr -> i32
// CHECK: llvm.add
    %9 = arith.addi %8, %4 : i32
// CHECK: llvm.sub
    %10 = arith.subi %6, %1 : i64
    llvm.br ^bb1(%9, %10 : i32, i64)
  ^bb3:  // pred: ^bb1
    llvm.store %5, %3 : i32, !ptr.ptr
// CHECK: omp.terminator
    omp.terminator
  }
  llvm.return
}

// -----

// CHECK-LABEL:   llvm.func @_QPtarget_map_with_bounds(
// CHECK:           %[[ARG_0:.*]]: !ptr.ptr, %[[ARG_1:.*]]: !ptr.ptr, %[[ARG_2:.*]]: !ptr.ptr) {
// CHECK: %[[C_01:.*]] = llvm.mlir.constant(4 : index) : i64
// CHECK: %[[C_02:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK: %[[C_03:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK: %[[C_04:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK: %[[BOUNDS0:.*]] = omp.bounds   lower_bound(%[[C_02]] : i64) upper_bound(%[[C_01]] : i64) stride(%[[C_04]] : i64) start_idx(%[[C_04]] : i64)
// CHECK: %[[MAP0:.*]] = omp.map_info var_ptr(%[[ARG_1]] : !ptr.ptr, !llvm.array<10 x i32>)   map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS0]]) -> !ptr.ptr {name = ""}
// CHECK: %[[C_11:.*]] = llvm.mlir.constant(4 : index) : i64
// CHECK: %[[C_12:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK: %[[C_13:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK: %[[C_14:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK: %[[BOUNDS1:.*]] = omp.bounds   lower_bound(%[[C_12]] : i64) upper_bound(%[[C_11]] : i64) stride(%[[C_14]] : i64) start_idx(%[[C_14]] : i64)
// CHECK: %[[MAP1:.*]] = omp.map_info var_ptr(%[[ARG_2]] : !ptr.ptr, !llvm.array<10 x i32>)   map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS1]]) -> !ptr.ptr {name = ""}
// CHECK: omp.target   map_entries(%[[MAP0]] -> %[[BB_ARG0:.*]], %[[MAP1]]  -> %[[BB_ARG1:.*]] : !ptr.ptr, !ptr.ptr) {
// CHECK: ^bb0(%[[BB_ARG0]]: !ptr.ptr, %[[BB_ARG1]]: !ptr.ptr):
// CHECK:   omp.terminator
// CHECK: }
// CHECK: llvm.return
// CHECK:}

llvm.func @_QPtarget_map_with_bounds(%arg0: !ptr.ptr, %arg1: !ptr.ptr, %arg2: !ptr.ptr) {
  %0 = llvm.mlir.constant(4 : index) : i64
  %1 = llvm.mlir.constant(1 : index) : i64
  %2 = llvm.mlir.constant(1 : index) : i64
  %3 = llvm.mlir.constant(1 : index) : i64
  %4 = omp.bounds   lower_bound(%1 : i64) upper_bound(%0 : i64) stride(%3 : i64) start_idx(%3 : i64)
  %5 = omp.map_info var_ptr(%arg1 : !ptr.ptr, !llvm.array<10 x i32>)   map_clauses(tofrom) capture(ByRef) bounds(%4) -> !ptr.ptr {name = ""}
  %6 = llvm.mlir.constant(4 : index) : i64
  %7 = llvm.mlir.constant(1 : index) : i64
  %8 = llvm.mlir.constant(1 : index) : i64
  %9 = llvm.mlir.constant(1 : index) : i64
  %10 = omp.bounds   lower_bound(%7 : i64) upper_bound(%6 : i64) stride(%9 : i64) start_idx(%9 : i64)
  %11 = omp.map_info var_ptr(%arg2 : !ptr.ptr, !llvm.array<10 x i32>)   map_clauses(tofrom) capture(ByRef) bounds(%10) -> !ptr.ptr {name = ""}
  omp.target   map_entries(%5 -> %arg3, %11 -> %arg4: !ptr.ptr, !ptr.ptr) {
    ^bb0(%arg3: !ptr.ptr, %arg4: !ptr.ptr):
    omp.terminator
  }
  llvm.return
}
