add_mlir_upstream_library(MLIRLoopLikeInterfaceTestPasses
  TestBlockInLoop.cpp

  EXCLUDE_FROM_LIBMLIR
  )
mlir_target_link_libraries(MLIRLoopLikeInterfaceTestPasses PUBLIC
  MLIRPass
  MLIRLoopLikeInterface
  MLIRFuncDialect
  )
