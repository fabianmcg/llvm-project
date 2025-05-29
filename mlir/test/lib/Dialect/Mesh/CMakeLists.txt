# Exclude tests from libMLIR.so
add_mlir_upstream_library(MLIRMeshTest
  TestOpLowering.cpp
  TestReshardingSpmdization.cpp
  TestSimplifications.cpp

  EXCLUDE_FROM_LIBMLIR
  )
mlir_target_link_libraries(MLIRMeshTest PUBLIC
  MLIRMeshDialect
  MLIRMeshTransforms
  MLIRPass
  MLIRRewrite
  MLIRTransformUtils
  )
