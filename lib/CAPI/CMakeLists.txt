add_mlir_public_c_api_library(BRUTUSCAPI
  Dialects.cpp
  LINK_LIBS PUBLIC
  MLIRJulia
)