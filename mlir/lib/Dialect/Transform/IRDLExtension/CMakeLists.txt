add_mlir_dialect_upstream_library(MLIRTransformDialectIRDLExtension
  IRDLExtension.cpp
  IRDLExtensionOps.cpp

  DEPENDS
  MLIRTransformDialectIRDLExtensionOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRTransformDialect
  MLIRIRDL
)
