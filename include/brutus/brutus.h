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
extern "C" {
#endif

typedef void (*ExecutionEngineFPtrResult)(void **);

void brutus_init(jl_module_t *brutus);
mlir::ModuleOp brutus_codegen_jlir(mlir::MLIRContext *context, jl_value_t *methods, jl_method_instance_t *entry_mi);
void brutus_canonicalize(mlir::MLIRContext *context, mlir::ModuleOp *module);
void brutus_lower_to_standard(mlir::MLIRContext *context, mlir::ModuleOp *module);
void brutus_lower_to_llvm(mlir::MLIRContext *context, mlir::ModuleOp *module);
ExecutionEngineFPtrResult brutus_create_execution_engine(mlir::MLIRContext *context, mlir::ModuleOp *module, std::string name);
ExecutionEngineFPtrResult brutus_codegen(jl_value_t *methods, jl_method_instance_t *entry_mi, char emit_fptr, char dump_flags);

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // BRUTUS_H
