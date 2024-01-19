add_mlir_public_c_api_library(BRUTUSCAPI
  Dialects.cpp
  LINK_LIBS PUBLIC
  MLIRJulia
)

add_mlir_aggregate(
  brutus-c
  SHARED
  EMBED_LIBS
  BRUTUSCAPI
  PUBLIC_LIBS
  ${Julia_LIBRARY}
  ${Julia_Internal_LIBRARY}
)

install(TARGETS brutus-c DESTINATION lib)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(brutus-c PRIVATE "-Wl,-exclude-libs,ALL")
else()
  if(NOT CMAKE_C_VISIBILITY_PRESET STREQUAL "hidden" OR NOT CMAKE_CXX_VISIBILITY_PRESET STREQUAL "hidden")
      message(STATUS "brutus-c on this platform exports all symbols. Recommend building with CMAKE_(C|CXX)_VISIBILITY_PRESET=hidden or implement filtering support.")
  endif()
endif()
