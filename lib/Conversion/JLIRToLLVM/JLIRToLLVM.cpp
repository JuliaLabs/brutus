#include "brutus/Dialect/Julia/JuliaOps.h"
#include "brutus/Conversion/JLIRToLLVM/JLIRToLLVM.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "juliapriv/julia_private.h"

using namespace mlir;
using namespace jlir;

struct JLIRToLLVMTypeConverter : public TypeConverter {
    LLVM::LLVMDialect *llvm_dialect;
    LLVM::LLVMType void_type;
    LLVM::LLVMType jlvalue;
    LLVM::LLVMType pjlvalue;

    JLIRToLLVMTypeConverter(MLIRContext *ctx)
        : llvm_dialect(ctx->getRegisteredDialect<LLVM::LLVMDialect>()),
          void_type(LLVM::LLVMType::getVoidTy(llvm_dialect)),
          jlvalue(LLVM::LLVMType::createStructTy(
                      llvm_dialect, Optional<StringRef>("jl_value_t"))),
          pjlvalue(jlvalue.getPointerTo()) {

        assert(llvm_dialect && "LLVM IR dialect is not registered");
        addConversion(
            [&](JuliaType jt) {
                return julia_type_to_llvm((jl_value_t*)jt.getDatatype()); });
    }

    LLVM::LLVMType julia_bitstype_to_llvm(jl_value_t *bt) {
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

    LLVM::LLVMType julia_struct_to_llvm(jl_value_t *jt) {
        // this function converts a Julia Type into the equivalent LLVM struct
        // use this where C-compatible (unboxed) structs are desired
        // use julia_type_to_llvm directly when you want to preserve Julia's
        // type semantics
        if (jt == (jl_value_t*)jl_bottom_type)
            return void_type;
        if (jl_is_primitivetype(jt))
            return julia_bitstype_to_llvm(jt);
        jl_datatype_t *jst = (jl_datatype_t*)jt;
        if (jl_is_structtype(jt)
            && !(jst->layout && jl_is_layout_opaque(jst->layout))) {
            // bool is_tuple = jl_is_tuple_type(jt);
            jl_svec_t *ftypes = jl_get_fieldtypes(jst);
            size_t ntypes = jl_svec_len(ftypes);
            if (ntypes == 0 || (jst->layout && jl_datatype_nbits(jst) == 0))
                return void_type;

            // TODO: actually handle structs
        }

        return pjlvalue; // prjlvalue?
    }

    LLVM::LLVMType julia_type_to_llvm(jl_value_t *jt) {
        // this function converts a Julia Type into the equivalent LLVM type

        // TODO: something special needs to happen for functions, which right
        //       now will just get turned into `void_type`

        if (jt == jl_bottom_type || jt == (jl_value_t*)jl_void_type)
            return void_type;
        if (jl_is_concrete_immutable(jt)) {
            if (jl_datatype_nbits(jt) == 0)
                return void_type;
            return julia_struct_to_llvm(jt);
        }

        return pjlvalue; // prjlvalue?
    }

    LLVM::LLVMType convertToLLVMType(Type t) {
        return convertType(t).dyn_cast_or_null<LLVM::LLVMType>();
    }
};

template <typename SourceOp>
struct OpAndTypeConversionPattern : OpConversionPattern<SourceOp> {
    JLIRToLLVMTypeConverter &lowering;

    OpAndTypeConversionPattern(MLIRContext *ctx,
                               JLIRToLLVMTypeConverter &lowering)
        : OpConversionPattern<SourceOp>(ctx), lowering(lowering) {}
};

// is there some template magic that would allow us to combine
// `ToLLVMOpPattern`, `ToUnaryLLVMOpPattern`, and `ToTernaryLLVMOpPattern`?

template <typename SourceOp, typename LLVMOp>
struct ToLLVMOpPattern : public OpAndTypeConversionPattern<SourceOp> {
    using OpAndTypeConversionPattern<SourceOp>::OpAndTypeConversionPattern;

    PatternMatchResult matchAndRewrite(SourceOp op,
                                       ArrayRef<Value> operands,
                                       ConversionPatternRewriter &rewriter) const override {
        static_assert(
            std::is_base_of<OpTrait::OneResult<SourceOp>, SourceOp>::value,
            "expected single result op");
        rewriter.replaceOpWithNewOp<LLVMOp>(
            op, this->lowering.convertToLLVMType(op.getType()), operands);
        return this->matchSuccess();
    }
};

template <typename SourceOp, typename LLVMOp>
struct ToUnaryLLVMOpPattern : public OpAndTypeConversionPattern<SourceOp> {
    using OpAndTypeConversionPattern<SourceOp>::OpAndTypeConversionPattern;

    PatternMatchResult matchAndRewrite(SourceOp op,
                                       ArrayRef<Value> operands,
                                       ConversionPatternRewriter &rewriter) const override {
        static_assert(
            std::is_base_of<OpTrait::OneResult<SourceOp>, SourceOp>::value,
            "expected single result op");
        assert(operands.size() == 1 && "expected unary operation");
        rewriter.replaceOpWithNewOp<LLVMOp>(
            op, this->lowering.convertToLLVMType(op.getType()), operands.front());
        return this->matchSuccess();
    }
};

template <typename SourceOp, typename LLVMOp>
struct ToTernaryLLVMOpPattern : public OpAndTypeConversionPattern<SourceOp> {
    using OpAndTypeConversionPattern<SourceOp>::OpAndTypeConversionPattern;

    PatternMatchResult matchAndRewrite(SourceOp op,
                                       ArrayRef<Value> operands,
                                       ConversionPatternRewriter &rewriter) const override {
        static_assert(
            std::is_base_of<OpTrait::OneResult<SourceOp>, SourceOp>::value,
            "expected single result op");
        assert(operands.size() == 3 && "expected ternary operation");
        rewriter.replaceOpWithNewOp<LLVMOp>(
            op, this->lowering.convertToLLVMType(op.getType()),
            operands[0],
            operands[1],
            operands[2]);
        return this->matchSuccess();
    }
};

template <typename SourceOp>
struct ToUndefOpPattern : public OpAndTypeConversionPattern<SourceOp> {
    using OpAndTypeConversionPattern<SourceOp>::OpAndTypeConversionPattern;

    PatternMatchResult matchAndRewrite(SourceOp op,
                                       ArrayRef<Value> operands,
                                       ConversionPatternRewriter &rewriter) const override {
        static_assert(
            std::is_base_of<OpTrait::OneResult<SourceOp>, SourceOp>::value,
            "expected single result op");
        rewriter.replaceOpWithNewOp<LLVM::UndefOp>(
            op, this->lowering.convertToLLVMType(op.getType()));
        return this->matchSuccess();
    }
};

template <typename SourceOp, typename CmpOp, typename Predicate, Predicate predicate>
struct ToCmpOpPattern : public OpAndTypeConversionPattern<SourceOp> {
    using OpAndTypeConversionPattern<SourceOp>::OpAndTypeConversionPattern;

    PatternMatchResult matchAndRewrite(SourceOp op,
                                       ArrayRef<Value> operands,
                                       ConversionPatternRewriter &rewriter) const override {
        assert(operands.size() == 2);
        CmpOp cmp = rewriter.create<CmpOp>(
            op.getLoc(), predicate, operands[0], operands[1]);
        rewriter.replaceOpWithNewOp<LLVM::ZExtOp>(
            op,
            this->lowering.convertToLLVMType(op.getResult().getType()),
            cmp.getResult());
        return this->matchSuccess();
    }
};

template <typename SourceOp, LLVM::ICmpPredicate predicate>
struct ToICmpOpPattern : public ToCmpOpPattern<SourceOp, LLVM::ICmpOp,
                                               LLVM::ICmpPredicate, predicate> {
    using ToCmpOpPattern<SourceOp, LLVM::ICmpOp,
                         LLVM::ICmpPredicate, predicate>::ToCmpOpPattern;
};

template <typename SourceOp, LLVM::FCmpPredicate predicate>
struct ToFCmpOpPattern : public ToCmpOpPattern<SourceOp, LLVM::FCmpOp,
                                               LLVM::FCmpPredicate, predicate> {
    using ToCmpOpPattern<SourceOp, LLVM::FCmpOp,
                         LLVM::FCmpPredicate, predicate>::ToCmpOpPattern;
};

struct FuncOpConversion : public OpAndTypeConversionPattern<FuncOp> {
    using OpAndTypeConversionPattern<FuncOp>::OpAndTypeConversionPattern;

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
        SmallVector<LLVM::LLVMType, 8> new_arg_types;
        new_arg_types.reserve(op.getNumArguments());
        SmallVector<std::pair<unsigned, Type>, 4> to_remove;
        TypeConverter::SignatureConversion result(op.getNumArguments());
        for (auto &en : llvm::enumerate(type.getInputs())) {
            LLVM::LLVMType converted =
                lowering.convertToLLVMType(en.value());
            assert(converted && "failed to convert argument type");

            // drop argument if it converts to void type
            if (converted == lowering.void_type) {
                // record that we need to remap it to an undef later, once we
                // have actually created the new function in which to add the
                // `LLVM::UndefOp`s
                to_remove.emplace_back(en.index(), converted);
                continue;
            }

            new_arg_types.push_back(converted);
            result.addInputs(en.index(), converted);
        }

        // create new function operation
        LLVM::LLVMType llvm_type = LLVM::LLVMType::getFunctionTy(
            new_return_type,
            new_arg_types,
            /*isVarArg=*/false);
        LLVM::LLVMFuncOp new_func = rewriter.create<LLVM::LLVMFuncOp>(
            op.getLoc(), op.getName(), llvm_type, LLVM::Linkage::External);
        rewriter.inlineRegionBefore(
            op.getBody(), new_func.getBody(), new_func.end());

        // insert `LLVM::UndefOp`s to start of new function to replace removed
        // arguments
        auto p = rewriter.saveInsertionPoint();
        rewriter.setInsertionPointToStart(&new_func.front());
        for (auto &entry : to_remove) {
            unsigned index = entry.first;
            Type converted = entry.second;
            LLVM::UndefOp replacement = rewriter.create<LLVM::UndefOp>(
                new_func.getArgument(index).getLoc(), // get location from old argument
                converted);
            result.remapInput(index, replacement.getResult());
        }
        rewriter.restoreInsertionPoint(p);

        rewriter.applySignatureConversion(&new_func.getBody(), result);
        rewriter.eraseOp(op);
        return matchSuccess();
    }
};

struct ConstantOpLowering : public OpAndTypeConversionPattern<ConstantOp> {
    using OpAndTypeConversionPattern<ConstantOp>::OpAndTypeConversionPattern;

    PatternMatchResult matchAndRewrite(ConstantOp op,
                                       ArrayRef<Value> operands,
                                       ConversionPatternRewriter &rewriter) const override {
        jl_value_t *julia_type = (jl_value_t*)op.getType().cast<JuliaType>().getDatatype();
        LLVM::LLVMType new_type = lowering.convertToLLVMType(op.getType());

        if (new_type == lowering.void_type) {
            rewriter.replaceOpWithNewOp<LLVM::UndefOp>(op, new_type);
            return matchSuccess();

        } else if (jl_is_primitivetype(julia_type)) {
            // TODO

        } else if (jl_is_structtype(julia_type)) {
            // TODO
        }

        if (new_type == lowering.pjlvalue) {
            LLVM::LLVMType int64 = LLVM::LLVMType::getInt64Ty(
                lowering.llvm_dialect);
            LLVM::ConstantOp address_op = rewriter.create<LLVM::ConstantOp>(
                op.getLoc(), int64, rewriter.getIntegerAttr(
                    rewriter.getIntegerType(64), (int64_t)op.value()));
            rewriter.replaceOpWithNewOp<LLVM::IntToPtrOp>(
                op, lowering.pjlvalue, address_op.getResult());
            // rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
            //     op, lowering.pjlvalue,
            //     rewriter.getIntegerAttr(rewriter.getIntegerType(64),
            //                             (int64_t)op.value()));
            return matchSuccess();
        }

        return matchFailure();
    }
};

struct CallOpLowering : public ToUndefOpPattern<CallOp> {
    // TODO
    using ToUndefOpPattern<CallOp>::ToUndefOpPattern;
};

struct InvokeOpLowering : public ToUndefOpPattern<InvokeOp> {
    // TODO
    using ToUndefOpPattern<InvokeOp>::ToUndefOpPattern;
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

struct GotoIfNotOpLowering : public OpAndTypeConversionPattern<GotoIfNotOp> {
    using OpAndTypeConversionPattern<GotoIfNotOp>::OpAndTypeConversionPattern;

    PatternMatchResult matchAndRewrite(GotoIfNotOp op,
                                       ArrayRef<Value> proper_operands,
                                       ArrayRef<Block *> destinations,
                                       ArrayRef<ArrayRef<Value>> operands,
                                       ConversionPatternRewriter &rewriter) const override {
        assert(proper_operands.size() == 1);
        assert(destinations.size() == 2 && operands.size() == 2);

        // truncate operand from i8 to i1
        LLVM::TruncOp truncated =
            rewriter.create<LLVM::TruncOp>(
                op.getLoc(),
                LLVM::LLVMType::getInt1Ty(lowering.llvm_dialect),
                proper_operands.front());

        rewriter.replaceOpWithNewOp<LLVM::CondBrOp>(
            op, truncated.getResult(), destinations,
            llvm::makeArrayRef({ValueRange(operands[0]),
                                ValueRange(operands[1])}));
        return matchSuccess();
    }
};

struct ReturnOpLowering : public OpAndTypeConversionPattern<ReturnOp> {
    using OpAndTypeConversionPattern<ReturnOp>::OpAndTypeConversionPattern;

    PatternMatchResult matchAndRewrite(ReturnOp op,
                                       ArrayRef<Value> operands,
                                       ConversionPatternRewriter &rewriter) const override {
        // drop operand if its type is the LLVM void type
        if (operands.size() == 1
            && operands.front().getType() == lowering.void_type) {
            operands = llvm::None;
        }
        rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, operands);
        return matchSuccess();
    }
};

struct PiOpLowering : public ToUndefOpPattern<PiOp> {
    // TODO
    using ToUndefOpPattern<PiOp>::ToUndefOpPattern;
};

struct JLIRToLLVMLoweringPass : public FunctionPass<JLIRToLLVMLoweringPass> {
    void runOnFunction() final {
        ConversionTarget target(getContext());
        target.addLegalDialect<LLVM::LLVMDialect>();

        OwningRewritePatternList patterns;
        JLIRToLLVMTypeConverter converter(&getContext());
        patterns.insert<
            FuncOpConversion,
            ToUndefOpPattern<UnimplementedOp>,
            ToUndefOpPattern<UndefOp>,
            ConstantOpLowering,
            CallOpLowering,
            InvokeOpLowering,
            GotoIfNotOpLowering,
            ReturnOpLowering,
            PiOpLowering,
            // bitcast
            // neg_int
            ToLLVMOpPattern<add_int, LLVM::AddOp>,
            ToLLVMOpPattern<sub_int, LLVM::SubOp>,
            ToLLVMOpPattern<mul_int, LLVM::MulOp>,
            ToLLVMOpPattern<sdiv_int, LLVM::SDivOp>,
            ToLLVMOpPattern<udiv_int, LLVM::UDivOp>,
            ToLLVMOpPattern<srem_int, LLVM::SRemOp>,
            ToLLVMOpPattern<urem_int, LLVM::URemOp>,
            // add_ptr
            // sub_ptr
            ToLLVMOpPattern<neg_float, LLVM::FNegOp>,
            ToLLVMOpPattern<add_float, LLVM::FAddOp>,
            ToLLVMOpPattern<sub_float, LLVM::FSubOp>,
            ToLLVMOpPattern<mul_float, LLVM::FMulOp>,
            ToLLVMOpPattern<div_float, LLVM::FDivOp>,
            ToLLVMOpPattern<rem_float, LLVM::FRemOp>,
            ToTernaryLLVMOpPattern<fma_float, LLVM::FMAOp>,
            // muladd_float
            // neg_float_fast
            // add_float_fast
            // sub_float_fast
            // mul_float_fast
            // div_float_fast
            // rem_float_fast
            ToICmpOpPattern<eq_int, LLVM::ICmpPredicate::eq>,
            ToICmpOpPattern<ne_int, LLVM::ICmpPredicate::ne>,
            ToICmpOpPattern<slt_int, LLVM::ICmpPredicate::slt>,
            ToICmpOpPattern<ult_int, LLVM::ICmpPredicate::ult>,
            ToICmpOpPattern<sle_int, LLVM::ICmpPredicate::sle>,
            ToICmpOpPattern<ule_int, LLVM::ICmpPredicate::ule>,
            ToFCmpOpPattern<eq_float, LLVM::FCmpPredicate::oeq>,
            ToFCmpOpPattern<ne_float, LLVM::FCmpPredicate::une>,
            ToFCmpOpPattern<lt_float, LLVM::FCmpPredicate::olt>,
            ToFCmpOpPattern<le_float, LLVM::FCmpPredicate::ole>,
            // fpiseq
            // fpislt
            ToLLVMOpPattern<and_int, LLVM::AndOp>,
            ToLLVMOpPattern<or_int, LLVM::OrOp>,
            ToLLVMOpPattern<xor_int, LLVM::XOrOp>,
            // not_int
            // shl_int
            // lshr_int
            // ashr_int
            // bswap_int
            // ctpop_int
            // ctlz_int
            // cttz_int
            // sext_int
            // zext_int
            // trunc_int
            // fptoui
            // fptosi
            // uitofp
            // sitofp
            // fptrunc
            // fpext
            // checked_sadd_int
            // checked_uadd_int
            // checked_ssub_int
            // checked_usub_int
            // checked_smul_int
            // checked_umul_int
            // checked_sdiv_int
            // checked_udiv_int
            // checked_srem_int
            // checked_urem_int
            ToUnaryLLVMOpPattern<abs_float, LLVM::FAbsOp>,
            // copysign_float
            // flipsign_int
            ToUnaryLLVMOpPattern<ceil_llvm, LLVM::FCeilOp>,
            // floor_llvm
            ToUnaryLLVMOpPattern<trunc_llvm, LLVM::TruncOp>,
            // rint_llvm
            ToUnaryLLVMOpPattern<sqrt_llvm, LLVM::SqrtOp>
            // sqrt_llvm_fast
            // pointerref
            // pointerset
            // cglobal
            // llvmcall
            // arraylen
            // cglobal_auto
            >(&getContext(), converter);
        patterns.insert<
            GotoOpLowering
            >(&getContext());

        if (failed(applyPartialConversion(
                       getFunction(), target, patterns, &converter)))
            signalPassFailure();
    }
};

std::unique_ptr<Pass> mlir::jlir::createJLIRToLLVMLoweringPass() {
  return std::make_unique<JLIRToLLVMLoweringPass>();
}
