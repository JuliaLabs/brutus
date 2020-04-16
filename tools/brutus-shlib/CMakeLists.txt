get_property(brutus_libs GLOBAL PROPERTY BRUTUS_LIBS)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
  MLIRIR
  ${dialect_libs}
  ${conversion_libs}
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

# TODO: check if this is needed (it was removed from MLIR)
function(whole_archive_link target)
  add_dependencies(${target} ${ARGN})
  if("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
    set(link_flags "-L${CMAKE_BINARY_DIR}/lib ")
    FOREACH(LIB ${ARGN})
      if("${CMAKE_GENERATOR}" STREQUAL "Xcode")
        string(CONCAT link_flags ${link_flags} "-Wl,-force_load ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib/lib${LIB}.a ")
      else()
        string(CONCAT link_flags ${link_flags} "-Wl,-force_load ${CMAKE_BINARY_DIR}/lib/lib${LIB}.a ")
      endif()
    ENDFOREACH(LIB)
  elseif(MSVC)
    FOREACH(LIB ${ARGN})
      string(CONCAT link_flags ${link_flags} "/WHOLEARCHIVE:${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib/${LIB}.lib ")
    ENDFOREACH(LIB)
  else()
    set(link_flags "-L${CMAKE_BINARY_DIR}/lib -Wl,--whole-archive,")
    FOREACH(LIB ${ARGN})
      string(CONCAT link_flags ${link_flags} "-l${LIB},")
    ENDFOREACH(LIB)
    string(CONCAT link_flags ${link_flags} "--no-whole-archive")
  endif()
  set_target_properties(${target} PROPERTIES LINK_FLAGS ${link_flags})
endfunction(whole_archive_link)

whole_archive_link(brutus ${brutus_libs} ${FULL_LIBS})

target_link_libraries(brutus
                      PRIVATE
                      ${Julia_LIBRARY}
                      ${brutus_libs}
                      ${FULL_LIBS}
                      ${LIBS})
