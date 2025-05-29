add_mlir_upstream_c_api_upstream_library(MLIRCAPITransforms
  Passes.cpp
  Rewrite.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRTransforms
  MLIRTransformUtils
)
