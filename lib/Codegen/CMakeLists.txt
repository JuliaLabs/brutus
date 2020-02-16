file(GLOB globbed *.c *.cpp)
add_llvm_library(BRUTUSCodegen
  ${globbed}

  ADDITIONAL_HEADER_DIRS
  ${BRUTUS_INCLUDE_DIR}/JuliaOps
  )
add_dependencies(BRUTUSCodegen
    MLIRJulia
)
target_link_libraries(BRUTUSCodegen
)
set_property(GLOBAL APPEND PROPERTY BRUTUS_LIBS BRUTUSCodegen)
