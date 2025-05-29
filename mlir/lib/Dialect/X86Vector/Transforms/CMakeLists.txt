add_mlir_dialect_upstream_library(MLIRX86VectorTransforms
  AVXTranspose.cpp
  LegalizeForLLVMExport.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRX86VectorDialect
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRVectorDialect
  MLIRVectorUtils
  )
