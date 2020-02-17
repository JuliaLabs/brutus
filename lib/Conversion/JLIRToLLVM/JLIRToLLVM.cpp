
#include "brutus/Dialect/Julia/JuliaOps.h"
#include "brutus/Conversion/JLIRToLLVM/JLIRToLLVM.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "juliapriv/julia_private.h"
using namespace mlir;
using namespace jlir;

LLVM::LLVMType bitstype_to_llvm(LLVM::LLVMDialect *llvm_dialect,
                                jl_value_t *bt) {
    assert(jl_is_primitivetype(bt));
    if (bt == (jl_value_t*)jl_bool_type)
        return LLVM::LLVMType::getInt8Ty(llvm_dialect);
    if (bt == (jl_value_t*)jl_int32_type)
        return LLVM::LLVMType::getInt32Ty(llvm_dialect);
    if (bt == (jl_value_t*)jl_int64_type)
        return LLVM::LLVMType::getInt64Ty(llvm_dialect);
    // if (llvmcall && (bt == (jl_value_t*)jl_float16_type))
    //     return LLVM::LLVMType::getHalfTy(llvm_dialect);
    if (bt == (jl_value_t*)jl_float32_type)
        return LLVM::LLVMType::getFloatTy(llvm_dialect);
    if (bt == (jl_value_t*)jl_float64_type)
        return LLVM::LLVMType::getDoubleTy(llvm_dialect);
    int nb = jl_datatype_size(bt);
    return LLVM::LLVMType::getIntNTy(llvm_dialect, nb * 8);
}

LLVM::LLVMType struct_to_llvm(LLVM::LLVMDialect *llvm_dialect,
                              jl_value_t *jt) {
    // this function converts a Julia Type into the equivalent LLVM struct
    // use this where C-compatible (unboxed) structs are desired
    // use type_to_llvm directly when you want to preserve Julia's type
    // semantics
    if (jt == (jl_value_t*)jl_bottom_type)
        return LLVM::LLVMType::getVoidTy(llvm_dialect);
    if (jl_is_primitivetype(jt))
        return bitstype_to_llvm(llvm_dialect, jt);
    // TODO: actually handle structs
    LLVM::LLVMType jlvalue = LLVM::LLVMType::createStructTy(
        llvm_dialect, Optional<StringRef>("jl_value_t"));
    LLVM::LLVMType pjlvalue = jlvalue.getPointerTo();
    return pjlvalue; // prjlvalue?
}

LLVM::LLVMType type_to_llvm(LLVM::LLVMDialect *llvm_dialect,
                            jl_value_t *jt) {
    // this function converts a Julia Type into the equivalent LLVM type
    if (jt == jl_bottom_type)
        return LLVM::LLVMType::getVoidTy(llvm_dialect);
    if (jl_is_concrete_immutable(jt)) {
        if (jl_datatype_nbits(jt) == 0)
            return LLVM::LLVMType::getVoidTy(llvm_dialect);
        return struct_to_llvm(llvm_dialect, jt);
    }

    LLVM::LLVMType jlvalue = LLVM::LLVMType::createStructTy(
        llvm_dialect, Optional<StringRef>("jl_value_t"));
    LLVM::LLVMType pjlvalue = jlvalue.getPointerTo();
    return pjlvalue; // prjlvalue?
}

struct JLIRToLLVMTypeConverter : public TypeConverter {
    LLVM::LLVMDialect *llvm_dialect;

    JLIRToLLVMTypeConverter(MLIRContext *ctx)
        : llvm_dialect(ctx->getRegisteredDialect<LLVM::LLVMDialect>()) {
        assert(llvm_dialect && "LLVM IR dialect is not registered");
    }

    Type convertType(Type t) final {
        JuliaType jt = t.cast<JuliaType>();
        return type_to_llvm(llvm_dialect, (jl_value_t*)jt.getDatatype());
    }

    LLVM::LLVMType convertToLLVMType(Type t) {
        return convertType(t).dyn_cast_or_null<LLVM::LLVMType>();
    }
};

struct FuncOpConversion : public OpConversionPattern<FuncOp> {
    JLIRToLLVMTypeConverter &lowering;

    FuncOpConversion(MLIRContext *ctx, JLIRToLLVMTypeConverter &lowering)
        : OpConversionPattern(ctx), lowering(lowering) {}

    PatternMatchResult matchAndRewrite(FuncOp op,
                                       ArrayRef<Value> operands,
                                       ConversionPatternRewriter &rewriter) const override {
        FunctionType type = op.getType();

        // convert return type
        assert(type.getNumResults() == 1);
        LLVM::LLVMType new_return_type =
            lowering.convertToLLVMType(type.getResults().front());
        assert(new_return_type && "failed to convert return type");

        // convert argument types
        TypeConverter::SignatureConversion result(op.getNumArguments());
        SmallVector<LLVM::LLVMType, 8> new_arg_types;
        new_arg_types.reserve(op.getNumArguments());
        for (auto &en : llvm::enumerate(type.getInputs())) {
            LLVM::LLVMType converted = lowering.convertToLLVMType(en.value());
            assert(converted && "failed to convert argument type");
            result.addInputs(en.index(), converted);
            new_arg_types.push_back(converted);
        }

        LLVM::LLVMType llvm_type = LLVM::LLVMType::getFunctionTy(
            new_return_type,
            new_arg_types,
            /*isVarArg=*/false);
        LLVM::LLVMFuncOp new_func = rewriter.create<LLVM::LLVMFuncOp>(
            op.getLoc(), op.getName(), llvm_type, LLVM::Linkage::External);

        rewriter.inlineRegionBefore(
            op.getBody(), new_func.getBody(), new_func.end());
        rewriter.applySignatureConversion(&new_func.getBody(), result);
        rewriter.eraseOp(op);
        return matchSuccess();
    }
};

struct ReturnOpLowering : public OpConversionPattern<ReturnOp> {
    using OpConversionPattern<ReturnOp>::OpConversionPattern;

    PatternMatchResult matchAndRewrite(ReturnOp op,
                                       ArrayRef<Value> operands,
                                       ConversionPatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, operands);
        return matchSuccess();
    }
};

struct GotoOpLowering : public OpConversionPattern<GotoOp> {
    using OpConversionPattern<GotoOp>::OpConversionPattern;

    PatternMatchResult matchAndRewrite(GotoOp op,
                                       ArrayRef<Value> proper_operands,
                                       ArrayRef<Block *> destinations,
                                       ArrayRef<ArrayRef<Value>> operands,
                                       ConversionPatternRewriter &rewriter) const override {
        assert(destinations.size() == 1 && operands.size() == 1);
        rewriter.replaceOpWithNewOp<LLVM::BrOp>(
            op, proper_operands, destinations,
            llvm::makeArrayRef(ValueRange(operands.front())));
        return matchSuccess();
    }
};

struct JLIRToLLVMLoweringPass : public FunctionPass<JLIRToLLVMLoweringPass> {
    void runOnFunction() final {
        ConversionTarget target(getContext());
        target.addLegalDialect<JLIRDialect, LLVM::LLVMDialect>();
        target.addIllegalOp<ReturnOp, GotoOp>();

        OwningRewritePatternList patterns;
        JLIRToLLVMTypeConverter converter(&getContext());
        patterns.insert<FuncOpConversion>(&getContext(), converter);
        patterns.insert<ReturnOpLowering, GotoOpLowering>(&getContext());

        if (failed(applyPartialConversion(
                       getFunction(), target, patterns, &converter)))
            signalPassFailure();
    }
};

std::unique_ptr<Pass> mlir::jlir::createJLIRToLLVMLoweringPass() {
  return std::make_unique<JLIRToLLVMLoweringPass>();
}