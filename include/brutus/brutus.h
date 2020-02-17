#ifndef BRUTUS_H
#define BRUTUS_H

#include "julia.h"

#ifdef __cplusplus
extern "C" {
#endif

void brutus_init();
void brutus_codegen(jl_value_t *ir_code, jl_value_t *ret_type, char *name, int optimize, int lower_to_llvm);

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // BRUTUS_H