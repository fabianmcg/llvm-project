set(LLVM_OPTIONAL_SOURCES
  MaskableOpInterface.cpp
  MaskingOpInterface.cpp
  )

function(add_mlir_interface_upstream_library name)
  add_mlir_upstream_library(MLIR${name}
    ${name}.cpp

    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Vector/Interfaces

    DEPENDS
    MLIR${name}IncGen

    LINK_LIBS PUBLIC
    MLIRIR
    )
endfunction(add_mlir_interface_upstream_library)

add_mlir_interface_upstream_library(MaskableOpInterface)
add_mlir_interface_upstream_library(MaskingOpInterface)
