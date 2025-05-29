add_mlir_dialect_upstream_library(MLIRTransformDebugExtension
  DebugExtension.cpp
  DebugExtensionOps.cpp

  DEPENDS
  MLIRTransformDialectDebugExtensionOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRTransformDialect
)
