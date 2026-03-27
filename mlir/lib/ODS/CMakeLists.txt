# This library contains the pure C++ ODS model classes (no LLVM TableGen
# dependency). It must not depend on MLIRTableGen or any LLVM TableGen headers.
llvm_add_library(MLIROds STATIC
  ODSContext.cpp
  Predicate.cpp

  DISABLE_LLVM_LINK_LLVM_DYLIB

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/ODS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Support
)
set_target_properties(MLIROds PROPERTIES FOLDER "MLIR/Tablegenning")

mlir_check_all_link_libraries(MLIROds)

add_mlir_library_install(MLIROds)
