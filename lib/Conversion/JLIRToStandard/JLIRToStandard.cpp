#include "brutus/Conversion/JLIRToStandard/JLIRToStandard.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/StandardTypes.h"

using namespace mlir;
using namespace jlir;

JLIRToStandardTypeConverter::JLIRToStandardTypeConverter(MLIRContext *ctx)
    : ctx(ctx) {

    // HACK: Until `materializeConversion` is called for 1-1 type conversions in
    //       MLIR, we convert function/block arguments to two result types,
    //       the second being a dummy `NoneType` type, to be removed in a second
    //       call to `applyPartialConversion`. The first two conversions are
    //       only necessary for this second call to `applyPartialConversion`.
    addConversion([](Type t) { return t; });
    addConversion([](NoneType t, SmallVectorImpl<Type> &results) {
        return success();
    });
    addConversion([this, ctx](JuliaType t, SmallVectorImpl<Type> &results) {
        llvm::Optional<Type> converted = convertJuliaType(t);
        if (converted.hasValue()) {
            results.push_back(converted.getValue());
            results.push_back(NoneType::get(ctx));
        } else {
            results.push_back(t);
        }
        return success();
    });
}

// returns `None` if the Julia type could not be converted to an MLIR builtin type
Optional<Type> JLIRToStandardTypeConverter::convertJuliaType(JuliaType t) {
    jl_datatype_t *jdt = t.getDatatype();
    if ((jl_value_t*)jdt == jl_bottom_type) {
        return None;
    } else if (jl_is_primitivetype(jdt)) {
        return convertBitstype(jdt);
    } else if (jl_is_structtype(jdt)
               && !(jdt->layout && jl_is_layout_opaque(jdt->layout))) {
        // bool is_tuple = jl_is_tuple_type(jt);
        jl_svec_t *ftypes = jl_get_fieldtypes(jdt);
        size_t ntypes = jl_svec_len(ftypes);
        if (ntypes == 0 || (jdt->layout && jl_datatype_nbits(jdt) == 0)) {
            return None;
        } else {
            // TODO: actually handle structs
            return t; // don't convert for now
        }
    }
    return None;
}

Type JLIRToStandardTypeConverter::convertBitstype(jl_datatype_t *jdt) {
    assert(jl_is_primitivetype(jdt));
    if (jdt == jl_bool_type) {
        // convert to i1 even though Julia converts to i8
        return IntegerType::get(1, ctx);
    } else if (jdt == jl_int32_type)
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
    // HACK
    assert(inputs.size() == 2 && inputs.back().getType().isa<NoneType>());

    return rewriter.create<ConvertStdOp>(loc, resultType, inputs.front());
}

namespace {

template <typename SourceOp>
struct OpAndTypeConversionPattern : OpConversionPattern<SourceOp> {
    JLIRToStandardTypeConverter &lowering;

    OpAndTypeConversionPattern(MLIRContext *ctx,
                               JLIRToStandardTypeConverter &lowering)
        : OpConversionPattern<SourceOp>(ctx), lowering(lowering) {}

    Optional<Value> convertValue(ConversionPatternRewriter &rewriter,
                                 Location location,
                                 Value originalValue,
                                 Value remappedOriginalValue) const {
        JuliaType type = originalValue.getType().cast<JuliaType>();
        Optional<Type> conversionResult =
            this->lowering.convertJuliaType(type);
        if (!conversionResult.hasValue())
            return None;
        ConvertStdOp convertOp = rewriter.create<ConvertStdOp>(
            location,
            conversionResult.getValue(),
            remappedOriginalValue);
        return convertOp.getResult();
    }

    LogicalResult
    convertOperands(ConversionPatternRewriter &rewriter,
                    Location location,
                    OperandRange originalOperands,
                    ArrayRef<Value> remappedOriginalOperands,
                    MutableArrayRef<Value> convertedOperands) const {
        unsigned i = 0;
        for (Value operand : originalOperands) {
            Optional<Value> conversionResult = this->convertValue(
                rewriter, location, operand, remappedOriginalOperands[i]);
            if (!conversionResult.hasValue())
                return failure();
            convertedOperands[i] = conversionResult.getValue();
            i++;
        }
        return success();
    }

    Value getIndexConstant(ConversionPatternRewriter &rewriter,
                           Location location,
                           int64_t value) const {
        auto op = rewriter.create<mlir::ConstantOp>(
            location,
            rewriter.getIndexType(),
            rewriter.getIntegerAttr(rewriter.getIndexType(), value));
        return op.getResult();
    }

    // assumes index is an integer type, not an index type
    Value decrementIndex(ConversionPatternRewriter &rewriter,
                         Location location,
                         Value index) const {
        // NOTE: this `ConvertStdOp` is used to convert from an MLIR integer to
        //       an index, unlike most other uses, which involve a Julia type
        ConvertStdOp convertedIndex = rewriter.create<ConvertStdOp>(
            location, rewriter.getIndexType(), index);
        SubIOp subOp = rewriter.create<SubIOp>(
            location,
            rewriter.getIndexType(),
            convertedIndex.getResult(),
            getIndexConstant(rewriter, location, 1));
        return subOp.getResult();
    }
};

template <typename SourceOp, typename StdOp>
struct ToStdOpPattern : public OpAndTypeConversionPattern<SourceOp> {
    using OpAndTypeConversionPattern<SourceOp>::OpAndTypeConversionPattern;

    LogicalResult matchAndRewrite(SourceOp op,
                                  ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        SmallVector<Value, 4> convertedOperands(operands.size());
        if (failed(this->convertOperands(
                       rewriter, op.getLoc(),
                       op.getOperation()->getOperands(), operands,
                       convertedOperands)))
            return failure();

        JuliaType returnType =
            op.getResult().getType().template cast<JuliaType>();
        Optional<Type> newReturnType =
            this->lowering.convertJuliaType(returnType);
        if (!newReturnType.hasValue())
            return failure();

        StdOp new_op = rewriter.create<StdOp>(
            op.getLoc(),
            newReturnType.getValue(),
            convertedOperands,
            None);
        rewriter.replaceOpWithNewOp<ConvertStdOp>(
            op, returnType, new_op.getResult());
        return success();
    }
};

template <typename SourceOp, typename CmpOp, typename Predicate, Predicate predicate>
struct ToCmpOpPattern : public OpAndTypeConversionPattern<SourceOp> {
    using OpAndTypeConversionPattern<SourceOp>::OpAndTypeConversionPattern;

    LogicalResult
    matchAndRewrite(SourceOp op,
                    ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const override {
        assert(operands.size() == 2);
        SmallVector<Value, 2> convertedOperands(operands.size());
        if (failed(this->convertOperands(
                       rewriter, op.getLoc(),
                       op.getOperands(), operands, convertedOperands)))
            return failure();

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
        Optional<Type> conversionResult = lowering.convertJuliaType(type);

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
        if (failed(convertOperands(rewriter, op.getLoc(), op.getOperands(),
                                   operands, convertedOperands)))
            return failure();
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
        if (failed(convertOperands(
                       rewriter,
                       op.getLoc(),
                       op.branchOperands(),
                       operands.slice(nProperOperands, nBranchOperands),
                       convertedBranchOperands)))
            return failure();
        if (failed(convertOperands(
                       rewriter,
                       op.getLoc(),
                       op.fallthroughOperands(),
                       operands.slice(nProperOperands + nBranchOperands,
                                      nFallthroughOperands),
                       convertedFallthroughOperands)))
            return failure();

        Optional<Value> conditionConversionResult = convertValue(
            rewriter, op.getLoc(), op.getOperand(0), operands.front());
        if (!conditionConversionResult.hasValue())
            return failure();

        rewriter.replaceOpWithNewOp<CondBranchOp>(
            op,
            conditionConversionResult.getValue(),
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
        Optional<Value> newOperand = convertValue(
            rewriter, op.getLoc(), op.getOperand(), operands.front());
        if (!newOperand.hasValue())
            return failure();

        rewriter.replaceOpWithNewOp<mlir::ReturnOp>(op, newOperand.getValue());
        return success();
    }
};

struct NotIntOpLowering : public OpAndTypeConversionPattern<Intrinsic_not_int> {
    using OpAndTypeConversionPattern<Intrinsic_not_int>::OpAndTypeConversionPattern;

    LogicalResult matchAndRewrite(Intrinsic_not_int op,
                                  ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        assert(operands.size() == 1);
        SmallVector<Value, 1> convertedOperands(operands.size());
        if (failed(convertOperands(rewriter, op.getLoc(), op.getOperands(),
                                   operands, convertedOperands)))
            return failure();

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

struct IsOpLowering : public OpAndTypeConversionPattern<Builtin_is> {
    using OpAndTypeConversionPattern<Builtin_is>::OpAndTypeConversionPattern;

    LogicalResult matchAndRewrite(Builtin_is op,
                                  ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        IntegerAttr falseAttr =
            rewriter.getIntegerAttr(rewriter.getI1Type(), 0);

        assert(operands.size() == 2);
        jl_value_t *jt1 = (jl_value_t*)op.getOperand(0).getType()
            .cast<JuliaType>().getDatatype();
        jl_value_t *jt2 = (jl_value_t*)op.getOperand(1).getType()
            .cast<JuliaType>().getDatatype();
        if (jl_is_concrete_type(jt1) && jl_is_concrete_type(jt2)
            && !jl_is_kind(jt1) && !jl_is_kind(jt2) && jt1 != jt2) {
            // disjoint concrete leaf types are never equal
            rewriter.replaceOpWithNewOp<mlir::ConstantOp>(op, falseAttr);
            return success();
        }

        // TODO: ghosts, see `emit_f_is`

        if (jl_type_intersection(jt1, jt2) == (jl_value_t*)jl_bottom_type) {
            rewriter.replaceOpWithNewOp<mlir::ConstantOp>(op, falseAttr);
            return success();
        }

        // TODO

        return failure();
    }
};

AffineExpr copied_makeCanonicalStridedLayoutExpr(ArrayRef<int64_t> sizes,
                                                ArrayRef<AffineExpr> exprs,
                                                MLIRContext *context) {
  AffineExpr expr;
  bool dynamicPoisonBit = false;
  unsigned numDims = 0;
  unsigned nSymbols = 0;
  llvm::outs() << "a: " << numDims << "\n";
  // Compute the number of symbols and dimensions of the passed exprs.
  for (AffineExpr expr : exprs) {
    expr.walk([&numDims, &nSymbols](AffineExpr d) {
                  if (AffineDimExpr dim = d.dyn_cast<AffineDimExpr>()) {
        llvm::outs() << "b: " << numDims << "\n";
        numDims = std::max(numDims, dim.getPosition() + 1);
        llvm::outs() << "c: " << numDims << "\n";
                  }
      else if (AffineSymbolExpr symbol = d.dyn_cast<AffineSymbolExpr>())
        nSymbols = std::max(nSymbols, symbol.getPosition() + 1);
    });
  }
  int64_t runningSize = 1;
  for (auto en : llvm::zip(llvm::reverse(exprs), llvm::reverse(sizes))) {
    int64_t size = std::get<1>(en);
    // Degenerate case, no size =-> no stride
    if (size == 0)
      continue;
    AffineExpr dimExpr = std::get<0>(en);
    AffineExpr stride = dynamicPoisonBit
                            ? getAffineSymbolExpr(nSymbols++, context)
                            : getAffineConstantExpr(runningSize, context);
    expr = expr ? expr + dimExpr * stride : dimExpr * stride;
    if (size > 0)
      runningSize *= size;
    else
      dynamicPoisonBit = true;
  }
  llvm::outs() << "findmehere: " << numDims << " " << nSymbols << "\n";
  exit(0);
  return simplifyAffineExpr(expr, numDims, nSymbols);
}

struct ArrayrefOpLowering : public OpAndTypeConversionPattern<Builtin_arrayref> {
    using OpAndTypeConversionPattern<Builtin_arrayref>::OpAndTypeConversionPattern;

    LogicalResult matchAndRewrite(Builtin_arrayref op,
                                  ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        JuliaType arrayType = op.getOperand(1).getType().cast<JuliaType>();
        jl_datatype_t *arrayDatatype = arrayType.getDatatype();
        assert(jl_is_array_type(arrayDatatype));

        JuliaType elementJuliaType = JuliaType::get(
            rewriter.getContext(), (jl_datatype_t*)jl_tparam0(arrayDatatype));
        Optional<Type> elementType = lowering.convertJuliaType(elementJuliaType);
        assert(elementType.hasValue() && "cannot convert Array element type");

        unsigned rank = jl_unbox_uint64(jl_tparam1(arrayDatatype));
        SmallVector<int64_t, 2> shape(rank, -1);

        SmallVector<Value, 2> indices;
        if (rank > 1 && op.getNumOperands() == 3) {
            // linear index, take advantage of lack of bounds checking
            indices.assign(
                rank, getIndexConstant(rewriter, op.getLoc(), 0));
            // change last index instead of first because MemRefs are row-major
            indices[rank-1] =
                decrementIndex(rewriter, op.getLoc(), operands[2]);
        } else {
            for (unsigned i = 2; i < op.getNumOperands(); i++) {
                indices.push_back(
                    decrementIndex(rewriter, op.getLoc(), operands[i]));
            }
        }

        // reverse order of dimensions because MemRefs are row-major whereas
        // Julia arrays are column-major
        SmallVector<unsigned, 2> permutation(
            llvm::reverse(llvm::seq<unsigned>(0, rank)));
        AffineMap affineMap = AffineMap::getPermutationMap(
            permutation, rewriter.getContext());
        // the following is only necessary because `mlir::getStridesAndOffset`
        // currently only supports `MemRef`s with a single affine map with a
        // single result
        copied_makeCanonicalStridedLayoutExpr(shape, affineMap.getResults(), rewriter.getContext()); // testing
        affineMap = AffineMap::get(
            rank, rank, makeCanonicalStridedLayoutExpr(
                shape, affineMap.getResults(), rewriter.getContext()));

        Value memref = rewriter.create<ArrayToMemRefOp>(
            op.getLoc(),
            MemRefType::get(shape, elementType.getValue(), affineMap),
            operands[1]).getResult();
        Value element = rewriter.create<LoadOp>(
            op.getLoc(), elementType.getValue(), memref, indices).getResult();
        rewriter.replaceOpWithNewOp<ConvertStdOp>(op, elementJuliaType, element);
        return success();
    }
};

} // namespace

bool JLIRToStandardLoweringPass::isFuncOpLegal(
    FuncOp op, JLIRToStandardTypeConverter &converter) {
    // function is illegal if any of its input or result types can but haven't
    // been converted
    FunctionType ft = op.getType().cast<FunctionType>();
    for (ArrayRef<Type> ts : {ft.getInputs(), ft.getResults()}) {
        for (Type t : ts) {
            if (JuliaType jt = t.dyn_cast<JuliaType>()) {
                if (converter.convertJuliaType(jt).hasValue())
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
    target.addLegalOp<ArrayToMemRefOp>();
    target.addLegalOp<UnimplementedOp>();
    target.addDynamicallyLegalOp<FuncOp>([this, &converter](FuncOp op) {
        return isFuncOpLegal(op, converter);
    });

    populateFuncOpTypeConversionPattern(patterns, &getContext(), converter);
    patterns.insert<
        // TESTING
        ArrayrefOpLowering,

        // ConvertStdOp    (JLIRToLLVM)
        // UnimplementedOp (JLIRToLLVM)
        // UndefOp         (JLIRToLLVM)
        ConstantOpLowering, // (also JLIRToLLVM)
        // CallOp
        // InvokeOp
        GotoOpLowering,
        GotoIfNotOpLowering, // (also JLIRToLLVM)
        ReturnOpLowering,    // (also JLIRToLLVM)
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
        // Intrinsic_fma_float (JLIRToLLVM)
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
        // Intrinsic_trunc_llvm (JLIRToLLVM, but maybe could be here?)
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
        IsOpLowering, // Builtin_is (also JLIRToLLVM)
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
        ToStdOpPattern<Builtin_ifelse, SelectOp> // (also JLIRToLLVM)
        // Builtin__typevar
        // invoke_kwsorter?
        >(&getContext(), converter);

    if (failed(applyPartialConversion(
                    getFunction(), target, patterns, &converter)))
        signalPassFailure();

    // HACK: Remove dummy `NoneType` function/block arguments. This is done
    //       using a second call to `applyPartialConversion` because block
    //       arguments for blocks in an op (such as the `FuncOp`) are only
    //       converted after all patterns have been applied (and only if
    //       legalization of the op succeeds).
    ConversionTarget hackTarget(getContext());
    hackTarget.addLegalDialect<StandardOpsDialect>();
    hackTarget.addLegalOp<ConvertStdOp>();
    hackTarget.addLegalOp<ArrayToMemRefOp>();
    hackTarget.addLegalOp<UnimplementedOp>();
    hackTarget.addDynamicallyLegalOp<FuncOp>([](FuncOp op) {
        // function is illegal if any of its input or result types are `NoneType`
        FunctionType ft = op.getType().cast<FunctionType>();
        for (ArrayRef<Type> ts : {ft.getInputs(), ft.getResults()}) {
            for (Type t : ts) {
                if (t.isa<NoneType>())
                    return false;
            }
        }
        return true;
    });
    OwningRewritePatternList hackPatterns;
    populateFuncOpTypeConversionPattern(hackPatterns, &getContext(), converter);
    if (failed(applyPartialConversion(
                    getFunction(), hackTarget, hackPatterns, &converter)))
        signalPassFailure();
}

std::unique_ptr<Pass> mlir::jlir::createJLIRToStandardLoweringPass() {
    return std::make_unique<JLIRToStandardLoweringPass>();
}
