add_mlir_dialect_library(
  MLIRJulia
  Ops.cpp
  ADDITIONAL_HEADER_DIRS
  StandaloneDialect.cpp
  StandaloneOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Brutus/JuliaOps

  DEPENDS
  MLIRJuliaOpsIncGen
  MLIRSideEffectInterfaces
  MLIRInferTypeOpInterface

  LINK_LIBS PUBLIC
  MLIR
)
llvm_update_compile_flags(MLIRJulia)
