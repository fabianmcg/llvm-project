add_mlir_translation_library(MLIRArithToLLVMIRTranslation
  ArithToLLVMIRTranslation.cpp
  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRDialectUtils
  MLIRIR
  MLIRArithDialect
  MLIRSupport
  MLIRTargetLLVMIRExport
  )
