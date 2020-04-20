#include "brutus/Dialect/Julia/JuliaOps.h"
#include "brutus/Conversion/JLIRToStandard/JLIRToStandard.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace jlir;

struct JLIRToStandardTypeConverter : public TypeConverter {
    MLIRContext *ctx;

    JLIRToStandardTypeConverter(MLIRContext *ctx) : ctx(ctx) {
        addConversion(
            [&](JuliaType t, SmallVectorImpl<Type> &results) {
                jl_datatype_t *jdt = t.getDatatype();
                if ((jl_value_t*)jdt == jl_bottom_type) {

                } else if (jl_is_primitivetype(jdt)) {
                    results.push_back(convert_bitstype(jdt));
                } else if (jl_is_structtype(jdt)
                           && !(jdt->layout && jl_is_layout_opaque(jdt->layout))) {
                    // bool is_tuple = jl_is_tuple_type(jt);
                    jl_svec_t *ftypes = jl_get_fieldtypes(jdt);
                    size_t ntypes = jl_svec_len(ftypes);
                    if (ntypes == 0 || (jdt->layout && jl_datatype_nbits(jdt) == 0)) {

                    } else {
                        // TODO: actually handle structs
                        results.push_back(t); // don't convert for now
                    }
                } else {
                    results.push_back(t); // don't convert other types
                }
                return success();
            });
    }

    Type convert_bitstype(jl_datatype_t *jdt) {
        assert(jl_is_primitivetype(jdt));
        if (jdt == jl_bool_type)
            return IntegerType::get(1, ctx); // will this work, or does it need to be 8?
        else if (jdt == jl_int32_type)
            return IntegerType::get(32, ctx);
        else if (jdt == jl_int64_type)
            return IntegerType::get(64, ctx);
        else if (jdt == jl_float32_type)
            return FloatType::getF32(ctx);
        else if (jdt == jl_float64_type)
            return FloatType::getF64(ctx);
        int nb = jl_datatype_size(jdt);
        return IntegerType::get(nb * 8, ctx);
    }
};

namespace {

template <typename SourceOp>
struct OpAndTypeConversionPattern : OpConversionPattern<SourceOp> {
    JLIRToStandardTypeConverter &lowering;

    OpAndTypeConversionPattern(MLIRContext *ctx,
                               JLIRToStandardTypeConverter &lowering)
        : OpConversionPattern<SourceOp>(ctx), lowering(lowering) {}
};

template <typename SourceOp, typename StdOp>
struct ToStdOpPattern : public OpAndTypeConversionPattern<SourceOp> {
    using OpAndTypeConversionPattern<SourceOp>::OpAndTypeConversionPattern;

    LogicalResult matchAndRewrite(SourceOp op,
                                  ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        static_assert(
            std::is_base_of<OpTrait::OneResult<SourceOp>, SourceOp>::value,
            "expected single result op");
        // MLIR doxygen says return types shouldn't change when using
        // `replaceOpWithNewOp`?
        rewriter.replaceOpWithNewOp<StdOp>(
            op, this->lowering.convertType(op.getType()), operands, llvm::None);
        return success();
    }
};

template <typename SourceOp, typename CmpOp, typename Predicate, Predicate predicate>
struct ToCmpOpPattern : public OpAndTypeConversionPattern<SourceOp> {
    using OpAndTypeConversionPattern<SourceOp>::OpAndTypeConversionPattern;

    LogicalResult matchAndRewrite(SourceOp op,
                                  ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        assert(operands.size() == 2);
        rewriter.replaceOpWithNewOp<CmpOp>(
            op, predicate, operands[0], operands[1]);
        return success();
    }
};

template <typename SourceOp, CmpIPredicate predicate>
struct ToCmpIOpPattern : public ToCmpOpPattern<SourceOp, CmpIOp,
                                               CmpIPredicate, predicate> {
    using ToCmpOpPattern<SourceOp, CmpIOp,
                         CmpIPredicate, predicate>::ToCmpOpPattern;
};

template <typename SourceOp, CmpFPredicate predicate>
struct ToCmpFOpPattern : public ToCmpOpPattern<SourceOp, CmpFOp,
                                               CmpFPredicate, predicate> {
    using ToCmpOpPattern<SourceOp, CmpFOp,
                         CmpFPredicate, predicate>::ToCmpOpPattern;
};

struct ConstantOpLowering : public OpAndTypeConversionPattern<jlir::ConstantOp> {
    using OpAndTypeConversionPattern<jlir::ConstantOp>::OpAndTypeConversionPattern;

    LogicalResult matchAndRewrite(jlir::ConstantOp op,
                                  ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        jl_value_t *value = op.value();
        jl_datatype_t *julia_type = op.getType().cast<JuliaType>().getDatatype();
        Type converted_type = lowering.convertType(op.getType());

        if (jl_is_primitivetype(julia_type)) {
            int nb = jl_datatype_size(julia_type);
            APInt val((julia_type == jl_bool_type) ? 1 : (8 * nb), 0);
            void *bits = const_cast<uint64_t*>(val.getRawData());
            assert(llvm::sys::IsLittleEndianHost);
            memcpy(bits, value, nb);

            Attribute value_attribute;
            if (FloatType ft = converted_type.dyn_cast<FloatType>()) {
                APFloat fval(ft.getFloatSemantics(), val);
                value_attribute = rewriter.getFloatAttr(ft, fval);
            } else {
                value_attribute = rewriter.getIntegerAttr(converted_type, val);
            }

            rewriter.replaceOpWithNewOp<mlir::ConstantOp>(
                op, value_attribute);
            return success();
        }

        return failure();
    }
};

struct GotoOpLowering : public OpAndTypeConversionPattern<GotoOp> {
    using OpAndTypeConversionPattern<GotoOp>::OpAndTypeConversionPattern;

    LogicalResult matchAndRewrite(GotoOp op,
                                  ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<BranchOp>(op, op.getSuccessor(), operands);
        return success();
    }
};

struct GotoIfNotOpLowering : public OpAndTypeConversionPattern<GotoIfNotOp> {
    using OpAndTypeConversionPattern<GotoIfNotOp>::OpAndTypeConversionPattern;

    LogicalResult matchAndRewrite(GotoIfNotOp op,
                                  ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<CondBranchOp>(
            op, operands.front(),
            op.falseDest(), op.fallthroughOperands(),
            op.trueDest(), op.branchOperands());
        return success();
    }
};

struct ReturnOpLowering : public OpAndTypeConversionPattern<jlir::ReturnOp> {
    using OpAndTypeConversionPattern<jlir::ReturnOp>::OpAndTypeConversionPattern;

    LogicalResult matchAndRewrite(jlir::ReturnOp op,
                                  ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<mlir::ReturnOp>(op, operands);
        return success();
    }
};

struct NotIntOpLowering : public OpAndTypeConversionPattern<Intrinsic_not_int> {
    using OpAndTypeConversionPattern<Intrinsic_not_int>::OpAndTypeConversionPattern;

    LogicalResult matchAndRewrite(Intrinsic_not_int op,
                                  ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        jl_datatype_t* operand_type =
            op.getOperand(0).getType().dyn_cast<JuliaType>().getDatatype();
        bool is_bool = operand_type == jl_bool_type;
        uint64_t mask_value = is_bool ? 1 : -1;
        // unsigned num_bits = 8 * (is_bool? 1 : jl_datatype_size(operand_type));
        unsigned num_bits = is_bool? 1 : (8*jl_datatype_size(operand_type));

        Value mask_constant =
            rewriter.create<mlir::ConstantOp>(
                op.getLoc(), operands.front().getType(),
                rewriter.getIntegerAttr(rewriter.getIntegerType(num_bits),
                                        // need APInt for sign extension
                                        APInt(num_bits, mask_value,
                                              /*isSigned=*/true)))
            .getResult();

        rewriter.replaceOpWithNewOp<XOrOp>(
            op, operands.front().getType(), operands.front(), mask_constant);
        return success();
    }
};

} // namespace

struct JLIRToStandardLoweringPass
    : public PassWrapper<JLIRToStandardLoweringPass, FunctionPass> {

    static bool isFuncOpLegal(Operation *op) {
        FunctionType ft = cast<FuncOp>(*op).getType().cast<FunctionType>();

        // for now, the function is illegal if any of its types
        // `jl_is_primitivetype`
        for (ArrayRef<Type> ts : {ft.getInputs(), ft.getResults()}) {
            for (Type t : ts) {
                if (JuliaType jt = t.dyn_cast_or_null<JuliaType>()) {
                    if (jl_is_primitivetype(jt.getDatatype()))
                        return false;
                }
            }
        }
        return true;
    }

    void runOnFunction() final {
        ConversionTarget target(getContext());
        target.addLegalDialect<StandardOpsDialect>();
        target.addDynamicallyLegalOp<FuncOp>(isFuncOpLegal);

        OwningRewritePatternList patterns;
        JLIRToStandardTypeConverter converter(&getContext());

        // TODO: what if return value is to be removed?
        // TODO: check that type conversion happens for block arguments
        populateFuncOpTypeConversionPattern(patterns, &getContext(), converter);
        patterns.insert<
            ConstantOpLowering,
            // CallOpLowering,
            // InvokeOpLowering,
            GotoOpLowering,
            GotoIfNotOpLowering,
            ReturnOpLowering,
            // PiOpLowering,
            // Intrinsic_bitcast
            // Intrinsic_neg_int
            ToStdOpPattern<Intrinsic_add_int, AddIOp>,
            ToStdOpPattern<Intrinsic_sub_int, SubIOp>,
            ToStdOpPattern<Intrinsic_mul_int, MulIOp>,
            ToStdOpPattern<Intrinsic_sdiv_int, SignedDivIOp>,
            ToStdOpPattern<Intrinsic_udiv_int, UnsignedDivIOp>,
            ToStdOpPattern<Intrinsic_srem_int, SignedRemIOp>,
            ToStdOpPattern<Intrinsic_urem_int, UnsignedRemIOp>,
            // Intrinsic_add_ptr
            // Intrinsic_sub_ptr
            ToStdOpPattern<Intrinsic_neg_float, NegFOp>,
            ToStdOpPattern<Intrinsic_add_float, AddFOp>,
            ToStdOpPattern<Intrinsic_sub_float, SubFOp>,
            ToStdOpPattern<Intrinsic_mul_float, MulFOp>,
            ToStdOpPattern<Intrinsic_div_float, DivFOp>,
            ToStdOpPattern<Intrinsic_rem_float, RemFOp>,
        //     ToTernaryLLVMOpPattern<Intrinsic_fma_float, LLVM::FMAOp>,
            // Intrinsic_muladd_float
            // Intrinsic_neg_float_fast
            // Intrinsic_add_float_fast
            // Intrinsic_sub_float_fast
            // Intrinsic_mul_float_fast
            // Intrinsic_div_float_fast
            // Intrinsic_rem_float_fast
            ToCmpIOpPattern<Intrinsic_eq_int, CmpIPredicate::eq>,
            ToCmpIOpPattern<Intrinsic_ne_int, CmpIPredicate::ne>,
            ToCmpIOpPattern<Intrinsic_slt_int, CmpIPredicate::slt>,
            ToCmpIOpPattern<Intrinsic_ult_int, CmpIPredicate::ult>,
            ToCmpIOpPattern<Intrinsic_sle_int, CmpIPredicate::sle>,
            ToCmpIOpPattern<Intrinsic_ule_int, CmpIPredicate::ule>,
            ToCmpFOpPattern<Intrinsic_eq_float, CmpFPredicate::OEQ>,
            ToCmpFOpPattern<Intrinsic_ne_float, CmpFPredicate::UNE>,
            ToCmpFOpPattern<Intrinsic_lt_float, CmpFPredicate::OLT>,
            ToCmpFOpPattern<Intrinsic_le_float, CmpFPredicate::OLE>,
            // Intrinsic_fpiseq
            // Intrinsic_fpislt
            ToStdOpPattern<Intrinsic_and_int, AndOp>,
            ToStdOpPattern<Intrinsic_or_int, OrOp>,
            ToStdOpPattern<Intrinsic_xor_int, XOrOp>,
            NotIntOpLowering, // Intrinsic_not_int
            ToStdOpPattern<Intrinsic_shl_int, ShiftLeftOp>,
            ToStdOpPattern<Intrinsic_lshr_int, UnsignedShiftRightOp>,
            ToStdOpPattern<Intrinsic_ashr_int, SignedShiftRightOp>,
            // Intrinsic_bswap_int
            // Intrinsic_ctpop_int
            // Intrinsic_ctlz_int
            // Intrinsic_cttz_int
            ToStdOpPattern<Intrinsic_sext_int, SignExtendIOp>,
            ToStdOpPattern<Intrinsic_zext_int, ZeroExtendIOp>,
            ToStdOpPattern<Intrinsic_trunc_int, TruncateIOp>,
            // Intrinsic_fptoui
            // Intrinsic_fptosi
            // Intrinsic_uitofp
            ToStdOpPattern<Intrinsic_sitofp, SIToFPOp>,
            ToStdOpPattern<Intrinsic_fptrunc, FPTruncOp>,
            ToStdOpPattern<Intrinsic_fpext, FPExtOp>,
            // Intrinsic_checked_sadd_int
            // Intrinsic_checked_uadd_int
            // Intrinsic_checked_ssub_int
            // Intrinsic_checked_usub_int
            // Intrinsic_checked_smul_int
            // Intrinsic_checked_umul_int
            // Intrinsic_checked_sdiv_int
            // Intrinsic_checked_udiv_int
            // Intrinsic_checked_srem_int
            // Intrinsic_checked_urem_int
            ToStdOpPattern<Intrinsic_abs_float, AbsFOp>,
            ToStdOpPattern<Intrinsic_copysign_float, CopySignOp>,
            // Intrinsic_flipsign_int
            ToStdOpPattern<Intrinsic_ceil_llvm, CeilFOp>,
            // Intrinsic_floor_llvm
        //     ToUnaryLLVMOpPattern<Intrinsic_trunc_llvm, LLVM::TruncOp>,
            // Intrinsic_rint_llvm
            ToStdOpPattern<Intrinsic_sqrt_llvm, SqrtOp>,
            // Intrinsic_sqrt_llvm_fast
            // Intrinsic_pointerref
            // Intrinsic_pointerset
            // Intrinsic_cglobal
            // Intrinsic_llvmcall
            // Intrinsic_arraylen
            // Intrinsic_cglobal_auto
            // Builtin_throw
        //     IsOpLowering, // Builtin_is
            // Builtin_typeof
            // Builtin_sizeof
            // Builtin_issubtype
        //     ToUndefOpPattern<Builtin_isa>, // Builtin_isa
            // Builtin__apply
            // Builtin__apply_pure
            // Builtin__apply_latest
            // Builtin__apply_iterate
            // Builtin_isdefined
            // Builtin_nfields
            // Builtin_tuple
            // Builtin_svec
        //     ToUndefOpPattern<Builtin_getfield>, // Builtin_getfield
            // Builtin_setfield
            // Builtin_fieldtype
            // Builtin_arrayref
            // Builtin_const_arrayref
            // Builtin_arrayset
            // Builtin_arraysize
            // Builtin_apply_type
            // Builtin_applicable
            // Builtin_invoke ?
            // Builtin__expr
            // Builtin_typeassert
            ToStdOpPattern<Builtin_ifelse, SelectOp>
            // Builtin__typevar
            // invoke_kwsorter?
            >(&getContext(), converter);

        if (failed(applyPartialConversion(
                       getFunction(), target, patterns, &converter)))
            signalPassFailure();
    }
};

std::unique_ptr<Pass> mlir::jlir::createJLIRToStandardLoweringPass() {
    return std::make_unique<JLIRToStandardLoweringPass>();
}
