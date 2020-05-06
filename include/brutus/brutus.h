#ifndef BRUTUS_H
#define BRUTUS_H

#include "julia.h"

#include "llvm-c/Types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*ExecutionEngineFPtrResult)(void **);

void brutus_init();
ExecutionEngineFPtrResult brutus_codegen(jl_value_t *ir_code,
                                         jl_value_t *ret_type,
                                         char *name,
                                         char emit_fptr,
                                         char optimize,
                                         char dump_flags);

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // BRUTUS_H
