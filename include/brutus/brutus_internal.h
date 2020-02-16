#ifndef BRUTUS_INTERNAL_H
#define BRUTUS_INTERNAL_H

// the difference between `brutus_internal.h` and `juliapriv/julia_internal.h` is about who initializes
// the data. Data in `brutus_internal.h` has to be initialized by us, whereas `juliapriv/julia_internal.h`
// is initialized by the julia runtime.

#include "julia.h"

#ifdef __cplusplus
extern "C" {
#endif

extern jl_sym_t *call_sym;    extern jl_sym_t *invoke_sym;

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // BRUTUS_INTERNAL_H