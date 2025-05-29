add_mlir_dialect_upstream_library(MLIREmitCTransforms
  Transforms.cpp
  FormExpressions.cpp
  TypeConversions.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/EmitC/Transforms

  DEPENDS
  MLIREmitCTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIREmitCDialect
  MLIRTransforms
)
