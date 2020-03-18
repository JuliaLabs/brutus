add_llvm_library(BRUTUSCodegen
  Brutus.cpp
  Codegen.cpp

  ADDITIONAL_HEADER_DIRS
  ${BRUTUS_INCLUDE_DIR}/JuliaOps
  )
add_dependencies(BRUTUSCodegen
    MLIRJulia
    MLIRTargetLLVMIR
)
target_link_libraries(BRUTUSCodegen
    MLIRTargetLLVMIR
)
set_property(GLOBAL APPEND PROPERTY BRUTUS_LIBS BRUTUSCodegen)
