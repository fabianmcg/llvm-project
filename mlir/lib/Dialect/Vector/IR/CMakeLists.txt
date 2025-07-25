add_mlir_dialect_library(MLIRVectorDialect
  VectorOps.cpp
  ValueBoundsOpInterfaceImpl.cpp
  ScalableValueBoundsConstraintSet.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Vector/IR

  DEPENDS
  MLIRMaskableOpInterfaceIncGen
  MLIRMaskingOpInterfaceIncGen
  MLIRVectorOpsIncGen
  MLIRVectorAttributesIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRArithDialect
  MLIRControlFlowInterfaces
  MLIRDataLayoutInterfaces
  MLIRDestinationStyleOpInterface
  MLIRDialectUtils
  MLIRIndexingMapOpInterface
  MLIRIR
  MLIRMaskableOpInterface
  MLIRMaskingOpInterface
  MLIRMemRefDialect
  MLIRSideEffectInterfaces
  MLIRTensorDialect
  MLIRUBDialect
  MLIRValueBoundsOpInterface
  MLIRVectorInterfaces
  )
