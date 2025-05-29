add_mlir_public_c_api_upstream_library(StandaloneCAPI
  Dialects.cpp
  LINK_LIBS PUBLIC
  MLIRStandalone
)
