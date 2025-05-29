add_mlir_upstream_library(MLIRPDLLCodeGen
  CPPGen.cpp
  MLIRGen.cpp

  LINK_LIBS PUBLIC
  MLIRParser
  MLIRPDLLAST
  MLIRPDLDialect
  MLIRSupport
  )
