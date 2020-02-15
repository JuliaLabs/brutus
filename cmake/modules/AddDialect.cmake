function(add_brutus_dialect dialect dialect_doc_filename)
  set(LLVM_TARGET_DEFINITIONS ${dialect}.td)
  mlir_tablegen(${dialect}.h.inc -gen-op-decls)
  mlir_tablegen(${dialect}.cpp.inc -gen-op-defs)
  add_public_tablegen_target(MLIR${dialect}IncGen)
  add_dependencies(brutus-headers MLIR${dialect}IncGen)

  # Generate Dialect Documentation
  set(LLVM_TARGET_DEFINITIONS ${dialect_doc_filename}.td)
  tablegen(MLIR ${dialect_doc_filename}.md -gen-op-doc "-I${MLIR_MAIN_SRC_DIR}" "-I${MLIR_INCLUDE_DIR}")
  set(GEN_DOC_FILE ${MLIR_BINARY_DIR}/docs/Dialects/${dialect_doc_filename}.md)
  add_custom_command(
          OUTPUT ${GEN_DOC_FILE}
          COMMAND ${CMAKE_COMMAND} -E copy
                  ${CMAKE_CURRENT_BINARY_DIR}/${dialect_doc_filename}.md
                  ${GEN_DOC_FILE}
          DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${dialect_doc_filename}.md)
  add_custom_target(${dialect_doc_filename}DocGen DEPENDS ${GEN_DOC_FILE})
  add_dependencies(brutus-doc ${dialect_doc_filename}DocGen)
endfunction()