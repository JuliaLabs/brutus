#ifndef BRUTUS_H
#define BRUTUS_H

#include "julia.h"

#ifdef __cplusplus
#include <cstdint>
extern "C" {
#else
#include <stdint.h>
#endif

void brutus_codegen(jl_value_t *ir_code, jl_value_t *ret_type, char *name, int optimize);

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // BRUTUS_H