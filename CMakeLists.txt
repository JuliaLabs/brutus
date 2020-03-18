cmake_minimum_required(VERSION 3.4.3)

# Based of SPIRV-LLVM Translator
set(BASE_LLVM_VERSION 11.0.0)
set(BRUTUS_VERSION ${BASE_LLVM_VERSION}.0)

if (NOT DEFINED BRUTUS_BUILD_EXTERNAL)
  # check if we build inside llvm or not
  if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(BRUTUS_BUILD_EXTERNAL YES)
  endif(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
endif (NOT DEFINED BRUTUS_BUILD_EXTERNAL)

# If we are not building as part of LLVM, build brutus as an
# standalone project using LLVM and MLIR as an external library
if(BRUTUS_BUILD_EXTERNAL)
  project(BRUTUS
    VERSION
      ${BRUTUS_VERSION}
    LANGUAGES
      CXX
  )

  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)

  find_package(LLVM ${BASE_LLVM_VERSION} REQUIRED CONFIG)
  list(APPEND CMAKE_MODULE_PATH
    ${LLVM_CMAKE_DIR}
  )
  include(AddLLVM)

  message(STATUS "${LLVM_CMAKE_DIR}")
  message(STATUS "Found LLVM: ${LLVM_VERSION}")

endif()

set(MLIR_TABLEGEN_EXE mlir-tblgen)

add_custom_target(brutus-headers)
set_target_properties(brutus-headers PROPERTIES FOLDER "Misc")
add_custom_target(brutus-doc)

# The next two variables need to be called this way for `mlir-tablegen` to work
set(MLIR_MAIN_SRC_DIR ${LLVM_EXTERNAL_MLIR_SOURCE_DIR})
set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include)
set(MLIR_BIN_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/../../tools/mlir/include)
set(BRUTUS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(BRUTUS_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${LLVM_EXTERNAL_MLIR_SOURCE_DIR}/cmake/modules")
include(AddMLIR)

list(APPEND CMAKE_MODULE_PATH "${BRUTUS_SOURCE_DIR}/cmake/modules")
include(AddDialect)
include(FindJulia)

set(BRUTUS_INCLUDE_DIR ${BRUTUS_BINARY_DIR}/include)

find_package(Julia REQUIRED)
# TODO: Check that Julia_LLVM_VERSION and this LLVM_VERSION match

include_directories("include")
include_directories(${Julia_INCLUDE_DIRS})
include_directories(${BRUTUS_INCLUDE_DIR})
include_directories(${MLIR_INCLUDE_DIR})
include_directories(${MLIR_BIN_INCLUDE_DIR})

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(tools)

if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  install(DIRECTORY include
    DESTINATION include
    COMPONENT brutus-headers
    FILES_MATCHING
    PATTERN "*.def"
    PATTERN "*.h"
    PATTERN "*.inc"
    PATTERN "*.td"
    PATTERN "LICENSE.TXT"
    )

  install(DIRECTORY ${MLIR_INCLUDE_DIR}
    DESTINATION include
    COMPONENT brutus-headers
    FILES_MATCHING
    PATTERN "*.def"
    PATTERN "*.h"
    PATTERN "*.gen"
    PATTERN "*.inc"
    PATTERN "*.td"
    PATTERN "CMakeFiles" EXCLUDE
    PATTERN "config.h" EXCLUDE
    )

  if (NOT LLVM_ENABLE_IDE)
    add_llvm_install_targets(install-brutus-headers
                             DEPENDS brutus-headers
                             COMPONENT brutus-headers)
  endif()
endif()
