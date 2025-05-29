add_mlir_dialect_upstream_library(MLIRUBDialect
  UBOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/UB

  DEPENDS
  MLIRUBOpsIncGen
  MLIRUBOpsInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  )
