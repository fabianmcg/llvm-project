add_mlir_dialect_upstream_library(MLIRAMXTransforms
  LegalizeForLLVMExport.cpp

  LINK_LIBS PUBLIC
  MLIRAMXDialect
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  )
