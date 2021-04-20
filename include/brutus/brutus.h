
#ifndef BRUTUS_H
#define BRUTUS_H

#include "julia.h"

#include "llvm-c/Types.h"
#include "mlir-c/IR.h"
#include <string>

#include "mlir/IR/Verifier.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

#ifdef __cplusplus
extern "C"
{
#endif
    void brutus_register_dialects(MlirContext Context);
    void brutus_register_extern_dialect(MlirContext Context, MlirDialect Dialect);
    MlirType brutus_get_juliatype(MlirContext context, jl_datatype_t *datatype);

    // Export C API for pipeline.
    typedef void (*ExecutionEngineFPtrResult)(void **);

    void brutus_init(jl_module_t *brutus);
    void brutus_codegen_jlir(MlirContext context, MlirModule module, jl_value_t *methods, jl_method_instance_t *entry_mi, char dump_flags);
    void brutus_canonicalize(MlirContext context, MlirModule module, char dump_flags);
    void brutus_lower_to_standard(MlirContext context, MlirModule module, char dump_flags);
    void brutus_lower_to_llvm(MlirContext context, MlirModule module, char dump_flags);
    ExecutionEngineFPtrResult brutus_create_execution_engine(MlirContext context, MlirModule module, std::string name);
    ExecutionEngineFPtrResult brutus_codegen(jl_value_t *methods, jl_method_instance_t *entry_mi, char emit_fptr, char dump_flags);

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // BRUTUS_H
