add_mlir_upstream_library(MLIROptLib
  MlirOptMain.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Tools/mlir-opt

  LINK_LIBS PUBLIC
  MLIRBytecodeWriter
  MLIRDebug
  MLIRObservers
  MLIRPass
  MLIRParser
  MLIRPluginsLib
  MLIRSupport
  MLIRIRDL
  )
