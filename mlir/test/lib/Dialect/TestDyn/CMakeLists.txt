add_mlir_dialect_upstream_library(MLIRTestDynDialect
  TestDynDialect.cpp

  EXCLUDE_FROM_LIBMLIR
)
mlir_target_link_libraries(MLIRTestDynDialect PUBLIC
  MLIRIR
)
