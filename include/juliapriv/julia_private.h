// This file is a part of Julia. License is MIT: https://julialang.org/license

#ifndef BRUTUS_JL_INTERNAL_H
#define BRUTUS_JL_INTERNAL_H

#ifdef __cplusplus
#include <cstdint>
extern "C" {
#else
#include <stdint.h>
#endif

/*****
 * Note: Only legal because these a JL_DLLEXPORT
 *****/

const char *jl_intrinsic_name(int f);
unsigned jl_intrinsic_nargs(int f);

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // BRUTUS_JL_INTERNAL_H