#ifndef BRUTUS_H
#define BRUTUS_H

#include "julia.h"

#include "llvm-c/Types.h"
#include "mlir-c/IR.h"
#include <string>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*ExecutionEngineFPtrResult)(void **);

void brutus_init(jl_module_t *brutus);
MlirContext *brutus_create_context(void);
MlirModule *brutus_codegen_jlir(MlirContext *context, jl_value_t *methods, jl_method_instance_t *entry_mi);
void brutus_canonicalize(MlirContext *context, MlirModule *module);
void brutus_lower_to_standard(MlirContext *context, MlirModule *module);
void brutus_lower_to_llvm(MlirContext *context, MlirModule *module);
ExecutionEngineFPtrResult brutus_create_execution_engine(MlirContext *context, MlirModule *module, std::string name);
ExecutionEngineFPtrResult brutus_codegen(jl_value_t *methods, jl_method_instance_t *entry_mi, char emit_fptr, char dump_flags);

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // BRUTUS_H
