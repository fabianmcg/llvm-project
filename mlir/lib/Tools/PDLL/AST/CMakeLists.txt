add_mlir_upstream_library(MLIRPDLLAST
  Context.cpp
  Diagnostic.cpp
  NodePrinter.cpp
  Nodes.cpp
  Types.cpp

  LINK_LIBS PUBLIC
  MLIRPDLLODS
  MLIRSupport
  )
