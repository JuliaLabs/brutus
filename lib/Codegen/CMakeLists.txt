add_mlir_library(
  BRUTUSCodegen
  Brutus.cpp
  Codegen.cpp

  ADDITIONAL_HEADER_DIRS
  ${BRUTUS_INCLUDE_DIR}/JuliaOps

  DEPENDS
  MLIRJuliaOpsIncGen

  LINK_LIBS PUBLIC
  MLIR
  # MLIRCAPIDebug
  # MLIRCAPIIR
  # MLIRCAPIRegistration  # TODO: See about dis-aggregating

  # Dialects
  # MLIRCAPILinalg  # TODO: Remove when above is removed.
  # MLIRCAPISparseTensor  # TODO: Remove when above is removed.
  # MLIRCAPIStandard
  # MLIRExecutionEngine
  # MLIRTargetLLVMIRExport

)
llvm_update_compile_flags(BRUTUSCodegen)
