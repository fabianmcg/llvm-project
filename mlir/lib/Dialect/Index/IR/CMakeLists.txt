add_mlir_dialect_upstream_library(MLIRIndexDialect
  IndexAttrs.cpp
  IndexDialect.cpp
  IndexOps.cpp
  InferIntRangeInterfaceImpls.cpp

  DEPENDS
  MLIRIndexOpsIncGen

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRIR
  MLIRCastInterfaces
  MLIRInferIntRangeCommon
  MLIRInferIntRangeInterface
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
  )
