// This file is a part of Julia. License is MIT: https://julialang.org/license

#ifndef BRUTUS_JL_INTERNAL_H
#define BRUTUS_JL_INTERNAL_H

#include "julia.h"

#ifdef __cplusplus
extern "C" {
#endif

/*****
 * Note: Only legal because these a JL_DLLEXPORT
 *****/

const char *jl_intrinsic_name(int f);
unsigned jl_intrinsic_nargs(int f);

#define jl_is_concrete_immutable(t) (jl_is_datatype(t) && (!((jl_datatype_t*)t)->mutabl) && ((jl_datatype_t*)t)->layout)

#define JL_CALLABLE(name)                                               \
    jl_value_t *name(jl_value_t *F, jl_value_t **args, uint32_t nargs)

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // BRUTUS_JL_INTERNAL_H
