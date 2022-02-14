configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

add_custom_target(precompile-brutus
    COMMAND
        ${Julia_EXECUTABLE} --project=${BRUTUS_SOURCE_DIR}/Brutus -e "using Pkg; Pkg.instantiate(); Pkg.precompile()"
    brutus
    VERBATIM
)

set(BRUTUS_TEST_DEPENDS
  FileCheck 
  brutus
  count
  precompile-brutus
  not
)


add_lit_testsuite(check-brutus "Running the brutus tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  #LIT ${LLVM_LIT}
  DEPENDS
  ${BRUTUS_TEST_DEPENDS}
  ARGS
  -a
  )

add_lit_testsuites(BRUTUS
  ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${BRUTUS_TEST_DEPENDS}
  )
