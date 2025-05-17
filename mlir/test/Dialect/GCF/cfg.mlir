// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(dump-op-cfg))" | FileCheck %s

// CHECK: digraph
func.func @cfg(%arg0 : index, %arg1 : index, %arg2 : index, %arg3 : i1) {
  scf.for %i0 = %arg0 to %arg1 step %arg2 {
    scf.for %i1 = %arg0 to %arg1 step %arg2 {
      %min_cmp = arith.cmpi slt, %i0, %i1 : index
      %min = arith.select %min_cmp, %i0, %i1 : index
      %max_cmp = arith.cmpi sge, %i0, %i1 : index
      %max = arith.select %max_cmp, %i0, %i1 : index
      scf.for %i2 = %min to %max step %i1 {
        scf.if %arg3 {
          %0 = index.add %arg1, %arg1
          gcf.if %arg3 {
            func.return
          }
        } else {
          %1 = index.add %arg1, %arg1
          gcf.for %gi0 = %arg0 to %arg1 step %arg2 {
            gcf.if %arg3 {
              gcf.break
            } else {
              gcf.continue
            }
            gcf.yield
          }
        }
      }
    }
  }
  return
}
