get_property(brutus_libs GLOBAL PROPERTY BRUTUS_LIBS)

set(LIBS
  MLIRIR
  MLIRAllDialects
  MLIRParser
  MLIRPass
  MLIRTranslation
  MLIRSupport
)

set(FULL_LIBS
  # MLIRTargetLLVMIR
  # MLIRTargetNVVMIR
)

add_llvm_library(brutus
                 SHARED
                 empty.cpp)

llvm_update_compile_flags(brutus)
whole_archive_link(brutus ${brutus_libs} ${FULL_LIBS})

target_link_libraries(brutus
                      PRIVATE
                      ${Julia_LIBRARY}
                      LLVM
                      ${brutus_libs}
                      ${FULL_LIBS}
                      ${LIBS})
