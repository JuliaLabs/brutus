#include "brutus/Conversion/JLIRToStandard/JLIRToStandard.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/StandardTypes.h"

using namespace mlir;
using namespace jlir;

JLIRToStandardTypeConverter::JLIRToStandardTypeConverter(MLIRContext *ctx)
    : ctx(ctx) {

    addConversion([this](JuliaType t) {
        return convert_JuliaType(t);
    });
}

Optional<Type> JLIRToStandardTypeConverter::convert_JuliaType(JuliaType t) {
    jl_datatype_t *jdt = t.getDatatype();
    if ((jl_value_t*)jdt == jl_bottom_type) {
        return llvm::None;
    } else if (jl_is_primitivetype(jdt)) {
        return convert_bitstype(jdt);
    // } else if (jl_is_structtype(jdt)
    //            && !(jdt->layout && jl_is_layout_opaque(jdt->layout))) {
    //     // bool is_tuple = jl_is_tuple_type(jt);
    //     jl_svec_t *ftypes = jl_get_fieldtypes(jdt);
    //     size_t ntypes = jl_svec_len(ftypes);
    //     if (ntypes == 0 || (jdt->layout && jl_datatype_nbits(jdt) == 0)) {

    //     } else {
    //         // TODO: actually handle structs
    //         results.push_back(t); // don't convert for now
    //     }
    }
    return llvm::None;
}

Type JLIRToStandardTypeConverter::convert_bitstype(jl_datatype_t *jdt) {
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

Operation
*JLIRToStandardTypeConverter::materializeConversion(PatternRewriter &rewriter,
                                                    Type resultType,
                                                    ArrayRef<Value> inputs,
                                                    Location loc) {
    return rewriter.create<ConvertStdOp>(loc, resultType, inputs.front());
}

namespace {

template <typename SourceOp>
struct OpAndTypeConversionPattern : OpConversionPattern<SourceOp> {
    JLIRToStandardTypeConverter &lowering;

    OpAndTypeConversionPattern(MLIRContext *ctx,
                               JLIRToStandardTypeConverter &lowering)
        : OpConversionPattern<SourceOp>(ctx), lowering(lowering) {}

    Value convertValue(ConversionPatternRewriter &rewriter,
                       Location location,
                       Value originalValue,
                       Value remappedOriginalValue,
                       bool relevant = false) const {
        JuliaType type = originalValue.getType().cast<JuliaType>();
        ConvertStdOp convertOp = rewriter.create<ConvertStdOp>(
            location,
            this->lowering.convert_JuliaType(type).getValue(),
            remappedOriginalValue);
        return convertOp.getResult();
    }

    void convertOperands(ConversionPatternRewriter &rewriter,
                         Location location,
                         OperandRange originalOperands,
                         ArrayRef<Value> remappedOriginalOperands,
                         MutableArrayRef<Value> convertedOperands) const {
        unsigned i = 0;
        for (Value operand : originalOperands) {
            convertedOperands[i] = this->convertValue(
                rewriter, location, operand, remappedOriginalOperands[i]);
            i++;
        }
    }
};

template <typename SourceOp, typename StdOp>
struct ToStdOpPattern : public OpAndTypeConversionPattern<SourceOp> {
    using OpAndTypeConversionPattern<SourceOp>::OpAndTypeConversionPattern;

    LogicalResult matchAndRewrite(SourceOp op,
                                  ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        SmallVector<Value, 4> convertedOperands(operands.size());
        this->convertOperands(
            rewriter, op.getLoc(),
            op.getOperation()->getOperands(), operands,
            convertedOperands);

        JuliaType returnType =
            op.getResult().getType().template cast<JuliaType>();
        StdOp new_op = rewriter.create<StdOp>(
            op.getLoc(),
            this->lowering.convert_JuliaType(returnType).getValue(),
            convertedOperands,
            None);
        rewriter.replaceOpWithNewOp<ConvertStdOp>(
            op, returnType, new_op.getResult());
        return success();
    }
};

// struct FuncOpConversion : public OpAndTypeConversionPattern<FuncOp> {
//     using OpAndTypeConversionPattern<FuncOp>::OpAndTypeConversionPattern;

//     Block* applyBlockConversion(ConversionPatternRewriter &rewriter,
//                                 Block *oldBlock) const {
//         llvm::outs() << "\n\n*********************applyBlockConversion\n\n";

//         // split block at beginning to obtain new block with no arguments
//         Block *newBlock = oldBlock->splitBlock(oldBlock->begin());

//         OpBuilder::InsertionGuard guard(rewriter);
//         rewriter.setInsertionPointToStart(newBlock);
//         for (BlockArgument &oldArgument : oldBlock->getArguments()) {
//             Type oldType = oldArgument.getType();
//             Type newType = lowering.convertType(oldType);
//             BlockArgument newArgument = newBlock->addArgument(newType);
//             if (oldType == newType) {
//                 rewriter.replaceUsesOfBlockArgument(oldArgument, newArgument);
//             } else {
//                 ConvertStdOp convertOp = rewriter.create<ConvertStdOp>(
//                     // is there a reasonable location for this? `BlockArgument`s
//                     // have unknown locations
//                     rewriter.getUnknownLoc(),
//                     oldType,
//                     newArgument);
//                 rewriter.replaceUsesOfBlockArgument(
//                     oldArgument, convertOp.getResult());
//             }
//         }

//         return newBlock;
//     }

//     // based on `FuncOpSignatureConversion::matchAndRewrite` in
//     // DialectConversion.cpp, except that `applyBlockConversion` (see above) is
//     // used in place of `ConversionPatternRewriter::applySignatureConversion` to
//     // convert type-converted block arguments back into `JuliaType`s with
//     // `ConvertStdOp`s, and this is applied to all blocks
//     LogicalResult
//     matchAndRewrite(FuncOp funcOp,
//                     ArrayRef<Value> operands,
//                     ConversionPatternRewriter &rewriter) const override {
//         FunctionType type = funcOp.getType();

//         // convert arguments
//         TypeConverter::SignatureConversion inputConversion(type.getNumInputs());
//         for (auto &en : llvm::enumerate(type.getInputs())) {
//             if (failed(lowering.convertSignatureArg(
//                            en.index(), en.value(), inputConversion)))
//                 return failure();
//         }

//         // convert results
//         SmallVector<Type, 1> convertedResults;
//         if (failed(lowering.convertTypes(type.getResults(), convertedResults)))
//             return failure();

//         // create new `FuncOp`
//         FunctionType newFuncType = rewriter.getFunctionType(
//             inputConversion.getConvertedTypes(), convertedResults);
//         FuncOp newFuncOp = rewriter.create<FuncOp>(
//             funcOp.getLoc(), funcOp.getName(), newFuncType, None);

//         // convert block argument types (replacing
//         // `rewriter.applySignatureConversion`)
//         for (Block &block : funcOp.getBlocks()) {
//             newFuncOp.getBody().push_back(
//                 applyBlockConversion(rewriter, &block));
//         }

//         rewriter.eraseOp(funcOp);

//         // rewriter.updateRootInPlace(funcOp, [&]() {

//         //     // instead of `rewriter.applySignatureConversion`
//         //     for (Block &block : funcOp.getBlocks())
//         //         newRegion.push_back(applyBlockConversion(rewriter, &block));

//         //     SmallVector<Block*, 4> oldBlocks;
//         //     for (Block &block : funcOp.getBlocks())
//         //         oldBlocks.push_back(&block);
//         //     for (Block *block : oldBlocks)
//         //         applyBlockConversion(rewriter, block);
//         // });

//         return success();
//     }
// };

template <typename SourceOp, typename CmpOp, typename Predicate, Predicate predicate>
struct ToCmpOpPattern : public OpAndTypeConversionPattern<SourceOp> {
    using OpAndTypeConversionPattern<SourceOp>::OpAndTypeConversionPattern;

    LogicalResult
    matchAndRewrite(SourceOp op,
                    ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const override {
        assert(operands.size() == 2);
        SmallVector<Value, 2> convertedOperands(2);
        this->convertOperands(
            rewriter, op.getLoc(),
            op.getOperands(), operands, convertedOperands);
        CmpOp cmpOp = rewriter.create<CmpOp>(
            op.getLoc(), predicate, convertedOperands[0], convertedOperands[1]);
        rewriter.replaceOpWithNewOp<ConvertStdOp>(
            op, op.getType(), cmpOp.getResult());
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
        JuliaType type = op.getType().cast<JuliaType>();
        jl_datatype_t *julia_type = type.getDatatype();
        Optional<Type> conversionResult = lowering.convert_JuliaType(type);

        if (!conversionResult.hasValue())
            return failure();

        Type convertedType = conversionResult.getValue();

        if (jl_is_primitivetype(julia_type)) {
            int nb = jl_datatype_size(julia_type);
            APInt val((julia_type == jl_bool_type) ? 1 : (8 * nb), 0);
            void *bits = const_cast<uint64_t*>(val.getRawData());
            assert(llvm::sys::IsLittleEndianHost);
            memcpy(bits, op.value(), nb);

            Attribute value_attribute;
            if (FloatType ft = convertedType.dyn_cast<FloatType>()) {
                APFloat fval(ft.getFloatSemantics(), val);
                value_attribute = rewriter.getFloatAttr(ft, fval);
            } else {
                value_attribute = rewriter.getIntegerAttr(convertedType, val);
            }

            mlir::ConstantOp constantOp = rewriter.create<mlir::ConstantOp>(
                op.getLoc(), value_attribute);
            rewriter.replaceOpWithNewOp<ConvertStdOp>(
                op, type, constantOp.getResult());
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
        SmallVector<Value, 4> convertedOperands(operands.size());
        convertOperands(
            rewriter, op.getLoc(), op.getOperands(), operands, convertedOperands);
        rewriter.replaceOpWithNewOp<BranchOp>(
            op, op.getSuccessor(), convertedOperands);
        return success();
    }
};

struct GotoIfNotOpLowering : public OpAndTypeConversionPattern<GotoIfNotOp> {
    using OpAndTypeConversionPattern<GotoIfNotOp>::OpAndTypeConversionPattern;

    LogicalResult matchAndRewrite(GotoIfNotOp op,
                                  ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        unsigned nBranchOperands = op.branchOperands().size();
        unsigned nFallthroughOperands = op.fallthroughOperands().size();
        unsigned nProperOperands =
            op.getNumOperands() - nBranchOperands - nFallthroughOperands;

        SmallVector<Value, 2> convertedBranchOperands(nBranchOperands);
        SmallVector<Value, 2> convertedFallthroughOperands(nFallthroughOperands);
        convertOperands(
            rewriter,
            op.getLoc(),
            op.branchOperands(),
            operands.slice(nProperOperands, nBranchOperands),
            convertedBranchOperands);
        convertOperands(
            rewriter,
            op.getLoc(),
            op.fallthroughOperands(),
            operands.slice(nProperOperands + nBranchOperands,
                           nFallthroughOperands),
            convertedFallthroughOperands);

        rewriter.replaceOpWithNewOp<CondBranchOp>(
            op,
            convertValue(
                rewriter, op.getLoc(), op.getOperand(0), operands.front()),
            op.fallthroughDest(), convertedFallthroughOperands,
            op.branchDest(), convertedBranchOperands);
        return success();
    }
};

struct ReturnOpLowering : public OpAndTypeConversionPattern<jlir::ReturnOp> {
    using OpAndTypeConversionPattern<jlir::ReturnOp>::OpAndTypeConversionPattern;

    LogicalResult matchAndRewrite(jlir::ReturnOp op,
                                  ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        Value oldOperand = op.getOperand();

        // ignore if operand is not a `JuliaType`
        if (!oldOperand.getType().isa<JuliaType>())
            return failure();

        rewriter.replaceOpWithNewOp<mlir::ReturnOp>(
            op,
            convertValue(rewriter, op.getLoc(), oldOperand, operands.front(), true));
        return success();
    }
};

struct NotIntOpLowering : public OpAndTypeConversionPattern<Intrinsic_not_int> {
    using OpAndTypeConversionPattern<Intrinsic_not_int>::OpAndTypeConversionPattern;

    LogicalResult matchAndRewrite(Intrinsic_not_int op,
                                  ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        // NOTE: this treats Bool as having been converted to i1

        assert(operands.size() == 1);
        SmallVector<Value, 1> convertedOperands(1);
        convertOperands(rewriter, op.getLoc(),
                        op.getOperands(), operands, convertedOperands);

        JuliaType oldType = op.getType().cast<JuliaType>();
        IntegerType newType =
            convertedOperands.front().getType().cast<IntegerType>();

        mlir::ConstantOp maskConstantOp =
            rewriter.create<mlir::ConstantOp>(
                op.getLoc(), newType,
                rewriter.getIntegerAttr(newType,
                                        // need APInt for sign extension
                                        APInt(newType.getWidth(), -1,
                                              /*isSigned=*/true)));

        XOrOp xorOp = rewriter.create<XOrOp>(
            op.getLoc(), newType,
            convertedOperands.front(), maskConstantOp.getResult());
        rewriter.replaceOpWithNewOp<ConvertStdOp>(
            op, oldType, xorOp.getResult());
        return success();
    }
};

struct JLIRToStandardHackTypeConverter : public JLIRToStandardTypeConverter {
    JLIRToStandardHackTypeConverter(MLIRContext *ctx)
        : JLIRToStandardTypeConverter(ctx) {

        // TODO: document hack
        addConversion([](Type t) { return t; }); // can this be moved to hack?
        addConversion([](NoneType t, SmallVectorImpl<Type> &results) {
            return success();
        });
        addConversion([this, ctx](JuliaType t, SmallVectorImpl<Type> &results) {
            llvm::Optional<Type> converted = convert_JuliaType(t);
            if (converted.hasValue()) {
                results.push_back(converted.getValue());
                results.push_back(NoneType::get(ctx));
            } else {
                results.push_back(t);
            }
            return success();
        });
    }
};

} // namespace

// HACK
namespace {

struct HackConversion : public OpAndTypeConversionPattern<FuncOp> {
    using OpAndTypeConversionPattern<FuncOp>::OpAndTypeConversionPattern;

    static bool hasNoneTypeArguments(FuncOp funcOp) {
        for (Block &block : funcOp.getBlocks()) {
            for (BlockArgument &blockArg : block.getArguments())
                if (blockArg.getType().isa<NoneType>())
                    return true;
        }
        return false;
    }

    LogicalResult
    matchAndRewrite(FuncOp funcOp,
                    ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const override {
        // should match if any block argument is of `NoneType`
        if (!hasNoneTypeArguments(funcOp))
            return failure();

        // the rest is pretty much straight from
        // `FuncOpSignatureConversion::matchAndRewrite`

        FunctionType type = funcOp.getType();

        // convert function arguments
        TypeConverter::SignatureConversion conversionResult(
            type.getNumInputs());
        for (auto &en : llvm::enumerate(type.getInputs())) {
            if (failed(lowering.convertSignatureArg(
                           en.index(), en.value(), conversionResult)))
                return failure();
        }

        // convert function results
        SmallVector<Type, 1> convertedResults;
        if (failed(lowering.convertTypes(type.getResults(), convertedResults)))
            return failure();

        // update function signature in-place
        rewriter.updateRootInPlace(
            funcOp,
            [&funcOp, &conversionResult, &convertedResults, &rewriter]() {
                funcOp.setType(FunctionType::get(
                                conversionResult.getConvertedTypes(),
                                convertedResults, funcOp.getContext()));
                rewriter.applySignatureConversion(
                    &funcOp.getBody(), conversionResult);
            });

        return success();
    }
};

} // namespace

bool JLIRToStandardLoweringPass::isFuncOpLegal(
    FuncOp op, JLIRToStandardTypeConverter &converter) {

    // HACK--the following will not be necessary once `materializeConversion`
    // for 1-1 type conversions has been implemented in MLIR. Note that block
    // arguments that are not also the function arguments are converted
    // automatically after all the patterns have been applied (block arguments
    // are not converted by any pattern), and this legality check seems not be
    // called for that, so the following in effect only detects if any of the
    // function arguments are of `NoneType`.
    if (HackConversion::hasNoneTypeArguments(op))
        return false;

    // function is illegal if any of its types can but haven't been
    // converted
    FunctionType ft = op.getType().cast<FunctionType>();
    for (ArrayRef<Type> ts : {ft.getInputs(), ft.getResults()}) {
        for (Type t : ts) {
            if (JuliaType jt = t.dyn_cast<JuliaType>()) {
                if (converter.convert_JuliaType(jt).hasValue())
                    return false;
            }
        }
    }
    return true;
}

void JLIRToStandardLoweringPass::runOnFunction() {
    ConversionTarget target(getContext());
    JLIRToStandardTypeConverter converter(&getContext());
    OwningRewritePatternList patterns;

    target.addLegalDialect<StandardOpsDialect>();
    target.addLegalOp<ConvertStdOp>();
    target.addDynamicallyLegalOp<FuncOp>([this, &converter](FuncOp op) {
        return isFuncOpLegal(op, converter);
    });

    // HACK--once `materializeConversion` for 1-1 type conversions has been
    // implemented in MLIR, the following can be replaced with a call to
    // `populateFuncOpTypeConversionPattern` with `converter` instead of
    // `hackConverter`.
    JLIRToStandardHackTypeConverter hackConverter(&getContext());
    populateFuncOpTypeConversionPattern(patterns, &getContext(), hackConverter);
    patterns.insert<HackConversion>(&getContext(), hackConverter);

    patterns.insert<
        // FuncOpConversion,
        ConstantOpLowering,
        // CallOp
        // InvokeOp
        GotoOpLowering,
        GotoIfNotOpLowering,
        ReturnOpLowering,
        // PiOp
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
        // Intrinsic_fma_float
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
        ToStdOpPattern<Intrinsic_sext_int, SignExtendIOp>, // TODO: args don't match
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
        // Intrinsic_trunc_llvm
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
        // Builtin_is
        // Builtin_typeof
        // Builtin_sizeof
        // Builtin_issubtype
        // Builtin_isa
        // Builtin__apply
        // Builtin__apply_pure
        // Builtin__apply_latest
        // Builtin__apply_iterate
        // Builtin_isdefined
        // Builtin_nfields
        // Builtin_tuple
        // Builtin_svec
        // Builtin_getfield
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

std::unique_ptr<Pass> mlir::jlir::createJLIRToStandardLoweringPass() {
    return std::make_unique<JLIRToStandardLoweringPass>();
}
