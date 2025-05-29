set(LLVM_OPTIONAL_SOURCES
  PythonTestCAPI.cpp
  PythonTestDialect.cpp
  PythonTestModulePybind11.cpp
  PythonTestModuleNanobind.cpp
)

add_mlir_upstream_library(MLIRPythonTestDialect
  PythonTestDialect.cpp

  EXCLUDE_FROM_LIBMLIR

  DEPENDS
  MLIRPythonTestIncGen
)
mlir_target_link_libraries(MLIRPythonTestDialect PUBLIC
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRSupport
)

add_mlir_public_c_api_upstream_library(MLIRCAPIPythonTestDialect
  PythonTestCAPI.cpp

  DEPENDS
  MLIRPythonTestIncGen

  LINK_LIBS PUBLIC
  MLIRCAPIInterfaces
  MLIRCAPIIR
  MLIRPythonTestDialect
)

