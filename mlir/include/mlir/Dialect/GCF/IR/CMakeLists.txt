add_mlir_dialect(GCFOps gcf Ops)
add_mlir_doc(GCFOps GCFDialect Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS GCFOps.td)
mlir_tablegen(GCFOpsAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=gcf)
mlir_tablegen(GCFOpsAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=gcf)
add_public_tablegen_target(MLIRGCFOpsAttributesIncGen)
