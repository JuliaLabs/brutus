file(GLOB globbed *.c *.cpp)
add_llvm_library(MLIRJulia
  ${globbed}

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
