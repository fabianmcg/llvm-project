// RUN: mlir-translate -test-to-cpp %s | FileCheck %s

// Exercises the external EmitC translation interfaces: DeclOpInterface,
// StmtOpInterface, ExprOpInterface and CxxTypeInterface, all implemented by
// ops/types in the test dialect.

// CHECK: template <typename T> struct Foo { T value; };
test.emitc_template_struct_decl "Foo"("T")

emitc.func @decl_stmts() {
  // CHECK-LABEL: void decl_stmts()
  // CHECK: int32_t x;
  test.emitc_decl_stmt "x" : i32
  // CHECK: Foo<int32_t> bar;
  test.emitc_decl_stmt "bar" : !test.emitc_template_inst<"Foo"<i32>>
  emitc.return
}

emitc.func @multi_arg_template() {
  // CHECK-LABEL: void multi_arg_template()
  // CHECK: pair<int32_t, double> p;
  test.emitc_decl_stmt "p" : !test.emitc_template_inst<"pair"<i32, f64>>
  emitc.return
}

emitc.func @member_call(%arg0 : !emitc.opaque<"Foo">) -> i32 {
  // CHECK-LABEL: int32_t member_call(Foo v1)
  // CHECK: int32_t v2 = 1;
  // CHECK: int32_t v3 = 2;
  // CHECK: int32_t v4 = v1.compute(v2, v3);
  // CHECK: return v4;
  %c1 = "emitc.constant"() {value = 1 : i32} : () -> i32
  %c2 = "emitc.constant"() {value = 2 : i32} : () -> i32
  %r = test.emitc_member_call_expr %arg0 "compute"(%c1, %c2)
      : (!emitc.opaque<"Foo">, i32, i32) -> i32
  emitc.return %r : i32
}
