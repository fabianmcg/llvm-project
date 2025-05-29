add_mlir_upstream_library(MLIRReduceLib
  MlirReduceMain.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRReduce
  MLIRSupport
  )
