#ifndef JLIR_CONVERSION_JLIRTOLLVM_JLIRTOLLVM_H_
#define JLIR_CONVERSION_JLIRTOLLVM_JLIRTOLLVM_H_

#include "brutus/Dialect/Julia/JuliaOps.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
// #include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
// #include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace jlir {

struct JLIRToLLVMTypeConverter : public LLVMTypeConverter {
    LLVM::LLVMDialect *llvmDialect;
    Type voidType;
    Type int1Type;
    Type int8Type;
    Type int16Type;
    Type int32Type;
    Type int64Type;
    Type sizeType;
    Type longType;
    Type mlirLongType;
    Type jlvalueType;
    Type pjlvalueType;
    Type jlarrayType;
    Type pjlarrayType;

    JLIRToLLVMTypeConverter(MLIRContext *ctx, LowerToLLVMOptions options);
    Type julia_bitstype_to_llvm(jl_value_t *bt);
    Type julia_struct_to_llvm(jl_value_t *jt);
    Type julia_type_to_llvm(jl_value_t *jt);
    Type INTT(Type t);
};

struct JLIRToLLVMLoweringPass
    : public PassWrapper<JLIRToLLVMLoweringPass, OperationPass<func::FuncOp>> {

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<LLVM::LLVMDialect>();
    }

    void runOnOperation() final;
};

/// Create a pass to convert JLIR operations to the LLVMIR dialect.
std::unique_ptr<Pass> createJLIRToLLVMLoweringPass();

} // namespace jlir
} // namespace mlir

#endif // JLIR_CONVERSION_JLIRTOLLVM_JLIRTOLLVM_H_
