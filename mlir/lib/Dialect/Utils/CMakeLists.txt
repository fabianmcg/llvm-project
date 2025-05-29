add_mlir_upstream_library(MLIRDialectUtils
  IndexingUtils.cpp
  ReshapeOpsUtils.cpp
  StructuredOpsUtils.cpp
  StaticValueUtils.cpp

  DEPENDS
  MLIRDialectUtilsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
