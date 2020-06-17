add_mlir_conversion_library(
  BRUTUSJLIRToStandard
  JLIRToStandard.cpp
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Brutus/JuliaOps
  LINK_LIBS PUBLIC
  MLIRJulia
)
llvm_update_compile_flags(BRUTUSJLIRToStandard)
