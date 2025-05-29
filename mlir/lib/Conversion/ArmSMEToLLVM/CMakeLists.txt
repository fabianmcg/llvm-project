add_mlir_conversion_upstream_library(MLIRArmSMEToLLVM
  ArmSMEToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/ArmSMEToLLVM

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArmSMETransforms
  MLIRArmSMEDialect
  MLIRTransforms
  MLIRLLVMCommonConversion
  MLIRLLVMDialect)
