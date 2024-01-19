
get_property(brutus_libs GLOBAL PROPERTY MLIR_STATIC_LIBS)

foreach (lib ${brutus_libs})
  if(XCODE)
    # Xcode doesn't support object libraries, so we have to trick it into
    # linking the static libraries instead.
    list(APPEND _DEPS "-force_load" ${lib})
  else()
    list(APPEND _OBJECTS $<TARGET_OBJECTS:obj.${lib}>)
  endif()
  list(APPEND _DEPS $<TARGET_PROPERTY:${lib},LINK_LIBRARIES>)
endforeach ()

llvm_add_library(
  brutus
  SHARED
  empty.cpp
  ${_OBJECTS}
  LINK_LIBS
  ${_DEPS}
  LLVM
  MLIR
  ${LLVM_PTHREAD_LIB}
  ${Julia_LIBRARY}
  ${Julia_Internal_LIBRARY}
)
llvm_update_compile_flags(brutus)
message("Libraries included in libBrutus.so: ${brutus_libs}")
# message("LLVM Components included in libMLIR.so: ${mlir_llvm_link_components}")

mlir_check_all_link_libraries(MLIR)
install(TARGETS brutus DESTINATION lib)