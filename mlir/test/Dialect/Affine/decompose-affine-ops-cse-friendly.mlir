// RUN: mlir-opt %s -allow-unregistered-dialect \
// RUN:   "-test-decompose-affine-ops=cse-friendly=true" \
// RUN:   -split-input-file | FileCheck %s --check-prefix=SHAPE

// RUN: mlir-opt %s -allow-unregistered-dialect \
// RUN:   "-test-decompose-affine-ops=cse-friendly=true" \
// RUN:   -split-input-file -cse | FileCheck %s --check-prefix=CSE

// SHAPE-DAG: #[[$times32:.*]] = affine_map<()[s0] -> (s0 * 32)>
// SHAPE-DAG: #[[$times16:.*]] = affine_map<()[s0] -> (s0 * 16)>
// SHAPE-DAG: #[[$times8:.*]] = affine_map<()[s0] -> (s0 * 8)>
// SHAPE-DAG: #[[$add:.*]] = affine_map<()[s0, s1] -> (s0 + s1)>
// SHAPE-DAG: #[[$c42:.*]] = affine_map<() -> (42)>
// SHAPE-DAG: #[[$c99:.*]] = affine_map<() -> (99)>

// CSE-DAG: #[[$times32:.*]] = affine_map<()[s0] -> (s0 * 32)>
// CSE-DAG: #[[$times16:.*]] = affine_map<()[s0] -> (s0 * 16)>
// CSE-DAG: #[[$times8:.*]] = affine_map<()[s0] -> (s0 * 8)>
// CSE-DAG: #[[$add:.*]] = affine_map<()[s0, s1] -> (s0 + s1)>
// CSE-DAG: #[[$c42:.*]] = affine_map<() -> (42)>
// CSE-DAG: #[[$c99:.*]] = affine_map<() -> (99)>

// SHAPE-LABEL: func.func @cse_friendly_shared_chain
// SHAPE-SAME:  %[[X:[0-9a-zA-Z]+]]: index,
// SHAPE-SAME:  %[[Y:[0-9a-zA-Z]+]]: index,
// SHAPE-SAME:  %[[Z:[0-9a-zA-Z]+]]: index

// CSE-LABEL: func.func @cse_friendly_shared_chain
// CSE-SAME:  %[[X:[0-9a-zA-Z]+]]: index,
// CSE-SAME:  %[[Y:[0-9a-zA-Z]+]]: index,
// CSE-SAME:  %[[Z:[0-9a-zA-Z]+]]: index
func.func @cse_friendly_shared_chain(%x: index, %y: index, %z: index) {

  // Shape test: In CSE-friendly mode, high-symbol operands are merged first.
  // For %a: chain is (z*32 + y*16) + x*8, then + 42.
  //
  // Chain for %a: starts from highest symbol, folds constant last.
  // SHAPE: %[[A_Z32:.*]] = affine.apply #[[$times32]]()[%[[Z]]]
  // SHAPE: %[[A_Y16:.*]] = affine.apply #[[$times16]]()[%[[Y]]]
  // SHAPE: %[[A_ZY:.*]] = affine.apply #[[$add]]()[%[[A_Z32]], %[[A_Y16]]]
  // SHAPE: %[[A_X8:.*]] = affine.apply #[[$times8]]()[%[[X]]]
  // SHAPE: %[[A_ZYX:.*]] = affine.apply #[[$add]]()[%[[A_ZY]], %[[A_X8]]]
  // SHAPE: %[[C42:.*]] = affine.apply #[[$c42]]()
  // SHAPE: %[[A:.*]] = affine.apply #[[$add]]()[%[[A_ZYX]], %[[C42]]]
  %a = affine.apply affine_map<()[s0, s1, s2] -> (42 + s0 * 8 + s1 * 16 + s2 * 32)>()[%x, %y, %z]

  // Chain for %b: identical intermediate chain, different constant.
  // SHAPE: %[[B_Z32:.*]] = affine.apply #[[$times32]]()[%[[Z]]]
  // SHAPE: %[[B_Y16:.*]] = affine.apply #[[$times16]]()[%[[Y]]]
  // SHAPE: %[[B_ZY:.*]] = affine.apply #[[$add]]()[%[[B_Z32]], %[[B_Y16]]]
  // SHAPE: %[[B_X8:.*]] = affine.apply #[[$times8]]()[%[[X]]]
  // SHAPE: %[[B_ZYX:.*]] = affine.apply #[[$add]]()[%[[B_ZY]], %[[B_X8]]]
  // SHAPE: %[[C99:.*]] = affine.apply #[[$c99]]()
  // SHAPE: %[[B:.*]] = affine.apply #[[$add]]()[%[[B_ZYX]], %[[C99]]]
  %b = affine.apply affine_map<()[s0, s1, s2] -> (99 + s0 * 8 + s1 * 16 + s2 * 32)>()[%x, %y, %z]

  // Common parts:
  // CSE: %[[Z32:.*]] = affine.apply #[[$times32]]()[%[[Z]]]
  // CSE: %[[Y16:.*]] = affine.apply #[[$times16]]()[%[[Y]]]
  // CSE: %[[ZY:.*]] = affine.apply #[[$add]]()[%[[Z32]], %[[Y16]]]
  // CSE: %[[X8:.*]] = affine.apply #[[$times8]]()[%[[X]]]
  // CSE: %[[ZYX:.*]] = affine.apply #[[$add]]()[%[[ZY]], %[[X8]]]
  //
  // Unique constant folds:
  // CSE: %[[C42:.*]] = affine.apply #[[$c42]]()
  // CSE: %[[A:.*]] = affine.apply #[[$add]]()[%[[ZYX]], %[[C42]]]
  // CSE: %[[C99:.*]] = affine.apply #[[$c99]]()
  // CSE: %[[B:.*]] = affine.apply #[[$add]]()[%[[ZYX]], %[[C99]]]

  // SHAPE: "some_side_effecting_consumer"(%[[A]]) : (index) -> ()
  // SHAPE: "some_side_effecting_consumer"(%[[B]]) : (index) -> ()
  // CSE: "some_side_effecting_consumer"(%[[A]]) : (index) -> ()
  // CSE: "some_side_effecting_consumer"(%[[B]]) : (index) -> ()
  "some_side_effecting_consumer"(%a) : (index) -> ()
  "some_side_effecting_consumer"(%b) : (index) -> ()
  return
}

// -----

// CSE-DAG: #[[$times16:.*]] = affine_map<()[s0] -> (s0 * 16)>
// CSE-DAG: #[[$div4:.*]] = affine_map<()[s0] -> (s0 floordiv 4)>
// CSE-DAG: #[[$add:.*]] = affine_map<()[s0, s1] -> (s0 + s1)>
// CSE-DAG: #[[$c100:.*]] = affine_map<() -> (100)>
// CSE-DAG: #[[$c200:.*]] = affine_map<() -> (200)>

// CSE-LABEL: func.func @cse_friendly_with_floordiv
// CSE-SAME:  %[[X:[0-9a-zA-Z]+]]: index,
// CSE-SAME:  %[[Y:[0-9a-zA-Z]+]]: index
func.func @cse_friendly_with_floordiv(%x: index, %y: index) {
  // CSE: %[[Y16:.*]] = affine.apply #[[$times16]]()[%[[Y]]]
  // CSE: %[[XDIV:.*]] = affine.apply #[[$div4]]()[%[[X]]]
  // CSE: %[[SHARED:.*]] = affine.apply #[[$add]]()[%[[Y16]], %[[XDIV]]]
  // CSE: %[[C100:.*]] = affine.apply #[[$c100]]()
  // CSE: %[[A:.*]] = affine.apply #[[$add]]()[%[[SHARED]], %[[C100]]]
  %a = affine.apply affine_map<()[s0, s1] -> (100 + s0 floordiv 4 + s1 * 16)>()[%x, %y]

  // CSE: %[[C200:.*]] = affine.apply #[[$c200]]()
  // CSE: %[[B:.*]] = affine.apply #[[$add]]()[%[[SHARED]], %[[C200]]]
  %b = affine.apply affine_map<()[s0, s1] -> (200 + s0 floordiv 4 + s1 * 16)>()[%x, %y]

  // CSE: "some_side_effecting_consumer"(%[[A]]) : (index) -> ()
  // CSE: "some_side_effecting_consumer"(%[[B]]) : (index) -> ()
  "some_side_effecting_consumer"(%a) : (index) -> ()
  "some_side_effecting_consumer"(%b) : (index) -> ()
  return
}
