func.func @main(%c: i1) {
  // Valid control-flow.
  test.region_cf r0 {
    test.to_r1
  } r1 {
    test.to_r0
  } r2 {
    test.to_parent
  }
  // Invalid control-flow according to the interface.
  test.region_cf r0 {
    // R0 is supposed to go to R1.
    test.to_r0
  } r1 {
    // R1 is supposed to go to R0.
    test.to_r1_or_r2
  } r2 {
    ^bb0:
    // Not valid, R2 is supposed to go to parent.
    cf.cond_br %c, ^bb1, ^bb2
    ^bb1:
    // This terminator says it can go to R0.
    test.to_r0
    ^bb2:
    test.to_parent
  }
  return
}
