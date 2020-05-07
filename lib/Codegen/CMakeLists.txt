add_mlir_library(
  BRUTUSCodegen
  Brutus.cpp
  Codegen.cpp

  ADDITIONAL_HEADER_DIRS
  ${BRUTUS_INCLUDE_DIR}/JuliaOps

  LINK_LIBS PUBLIC
  MLIR
)
llvm_update_compile_flags(BRUTUSCodegen)
