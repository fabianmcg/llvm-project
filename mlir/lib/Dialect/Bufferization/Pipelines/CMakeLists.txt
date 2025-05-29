add_mlir_dialect_upstream_library(MLIRBufferizationPipelines
  BufferizationPipelines.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Bufferization

  LINK_LIBS PUBLIC
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRMemRefTransforms
  MLIRFuncDialect
  MLIRPass
  MLIRTransforms
)
