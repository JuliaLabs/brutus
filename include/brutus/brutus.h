#ifndef BRUTUS_H
#define BRUTUS_H

#include "julia.h"

#include "llvm-c/Types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*ExecutionEngineFPtrResult)(void **);

void brutus_init();
ExecutionEngineFPtrResult brutus_codegen(jl_value_t *methods,
                                         jl_method_instance_t *entry_mi,
                                         char emit_fptr,
                                         char dump_flags);

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // BRUTUS_H
