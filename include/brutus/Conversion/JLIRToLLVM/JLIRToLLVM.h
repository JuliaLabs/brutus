#ifndef JLIR_CONVERSION_JLIRTOLLVM_JLIRTOLLVM_H_
#define JLIR_CONVERSION_JLIRTOLLVM_JLIRTOLLVM_H_

#include "brutus/Dialect/Julia/JuliaOps.h"

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace jlir {

struct JLIRToLLVMTypeConverter : public LLVMTypeConverter {
    LLVM::LLVMDialect *llvm_dialect;
    LLVM::LLVMType void_type;
    LLVM::LLVMType jlvalue;
    LLVM::LLVMType pjlvalue;

    JLIRToLLVMTypeConverter(MLIRContext *ctx);
    LLVM::LLVMType julia_bitstype_to_llvm(jl_value_t *bt);
    LLVM::LLVMType julia_struct_to_llvm(jl_value_t *jt);
    LLVM::LLVMType julia_type_to_llvm(jl_value_t *jt);
    LLVM::LLVMType INTT(LLVM::LLVMType t);
};

struct JLIRToLLVMLoweringPass
    : public PassWrapper<JLIRToLLVMLoweringPass, FunctionPass> {

    void runOnFunction() final;
};

/// Create a pass to convert JLIR operations to the LLVMIR dialect.
std::unique_ptr<Pass> createJLIRToLLVMLoweringPass();

} // namespace jlir
} // namespace mlir

#endif // JLIR_CONVERSION_JLIRTOLLVM_JLIRTOLLVM_H_
