add_mlir_conversion_upstream_library(MLIRArithAttrToLLVMConversion
  AttrToLLVMConverter.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRLLVMDialect
  )
