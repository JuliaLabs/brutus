add_llvm_library(BRUTUSJLIRToStandard
  JLIRToStandard.cpp
  ADDITIONAL_HEADER_DIRS
  ${BRUTUS_INCLUDE_DIR}/JuliaOps
  )
add_dependencies(BRUTUSJLIRToStandard
    MLIRJulia
)
target_link_libraries(BRUTUSJLIRToStandard
)
set_property(GLOBAL APPEND PROPERTY BRUTUS_LIBS BRUTUSJLIRToStandard)
