add_llvm_library(MLIRJulia
  Ops.cpp
  CallOp.cpp
  ADDITIONAL_HEADER_DIRS
  ${BRUTUS_INCLUDE_DIR}/JuliaOps
  )
add_dependencies(MLIRJulia
  MLIRIR
  MLIRJuliaOpsIncGen
  LLVMSupport
)
target_link_libraries(MLIRJulia
  MLIRIR
  LLVMSupport
)
set_property(GLOBAL APPEND PROPERTY BRUTUS_LIBS MLIRJulia)
