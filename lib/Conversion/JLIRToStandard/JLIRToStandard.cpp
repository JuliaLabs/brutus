#include "brutus/Conversion/JLIRToStandard/JLIRToStandard.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/StandardTypes.h"

using namespace mlir;
using namespace jlir;

JLIRToStandardTypeConverter::JLIRToStandardTypeConverter(MLIRContext *ctx)
    : ctx(ctx) {

    addConversion([this, ctx](JuliaType t, SmallVectorImpl<Type> &results) {
        // TODO: Drop ghosts?
        llvm::Optional<Type> converted = convertJuliaType(t);
        if (converted.hasValue()) {
            results.push_back(converted.getValue());
        } else {
            results.push_back(t);
        }
        return success();
    });

    // Materialize the cast for one-to-one conversions.
    addMaterialization([&](PatternRewriter &rewriter,
                           Type resultType, ValueRange inputs,
                           Location loc) -> Optional<Value> {
        if (inputs.size() == 1) {
            Value in = inputs.front();
            ConvertStdOp op = rewriter.create<ConvertStdOp>(loc, resultType, in);
            return op.getResult();
        } else {
            return None;
        }
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

namespace {

template <typename SourceOp>
struct JLIRToStdConversionPattern : OpConversionPattern<SourceOp> {
    JLIRToStdConversionPattern(MLIRContext *ctx,
                               JLIRToStandardTypeConverter &lowering)
        : OpConversionPattern<SourceOp>(lowering, ctx){}

    Value 
    convertValue(ConversionPatternRewriter &rewriter,
                Location location,
                Value value) const {

        auto converter = this->typeConverter;
        JuliaType from = value.getType().template dyn_cast<JuliaType>();
        if (from) {
            Type to = converter->convertType(from);
            if (to) {
                ConvertStdOp op = rewriter.create<ConvertStdOp>(location, to, value);
                return op.getResult();
            }
        }
        // passthrough
        return value;
    }

    void
    convertOperands(ConversionPatternRewriter &rewriter,
                    Location location,
                    ValueRange operands,
                    MutableArrayRef<Value> newOperands) const {
        unsigned i = 0;
        for (Value operand : operands) {
            newOperands[i] = this->convertValue(rewriter, location, operand);
            i++;
        }
    }

    Optional<Value>
    convertArray(ConversionPatternRewriter &rewriter,
                 Location loc,
                 Value value) const {
        auto converter = this->typeConverter;
        JuliaType arrayType = value.getType().template dyn_cast<JuliaType>();

        if (!arrayType)
            return None;
        
        jl_datatype_t *arrayDatatype = arrayType.getDatatype();

        if (!jl_is_array_type(arrayDatatype))
            return None;

        JuliaType elJLType = JuliaType::get(
            rewriter.getContext(), (jl_datatype_t*)jl_tparam0(arrayDatatype));
        Type elType = converter->convertType(elJLType);
        if (!elType) {
            failure();
        }
        unsigned rank = jl_unbox_uint64(jl_tparam1(arrayDatatype));
        SmallVector<int64_t, 2> shape(rank, -1);

        ArrayToMemRefOp memref = rewriter.create<ArrayToMemRefOp>(
            loc,
            MemRefType::get(shape, elType),
            value);
        return memref.getResult();
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
struct ToStdOpPattern : public JLIRToStdConversionPattern<SourceOp> {
    using JLIRToStdConversionPattern<SourceOp>::JLIRToStdConversionPattern;

    LogicalResult matchAndRewrite(SourceOp op,
                                    ArrayRef<Value> operands,
                                    ConversionPatternRewriter &rewriter) const override {
        auto result = this->typeConverter->convertType(op.getType());
        if (!result)
            return failure();

        SmallVector<Value, 4> newOperands(operands.size());
        this->convertOperands(rewriter, op.getLoc(), operands, newOperands);

        StdOp target = rewriter.create<StdOp>(op.getLoc(), result, newOperands, op.getAttrs());
        rewriter.replaceOpWithNewOp<ConvertStdOp>(op, op.getResult().getType(), target.getResult());

        return success();
    }
};

template <typename SourceOp, typename CmpOp, typename Predicate, Predicate predicate>
struct ToCmpOpPattern : public JLIRToStdConversionPattern<SourceOp> {
    using JLIRToStdConversionPattern<SourceOp>::JLIRToStdConversionPattern;

    LogicalResult
    matchAndRewrite(SourceOp op,
                    ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const override {
        assert(operands.size() == 2);
        SmallVector<Value, 2> newOperands(operands.size());
        this->convertOperands(rewriter, op.getLoc(), operands, newOperands);

        CmpOp target = rewriter.create<CmpOp>(op.getLoc(), predicate, newOperands[0], newOperands[1]);
        rewriter.replaceOpWithNewOp<ConvertStdOp>(op, op.getResult().getType(), target.getResult());
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

struct ConstantOpLowering : public JLIRToStdConversionPattern<jlir::ConstantOp> {
    using JLIRToStdConversionPattern<jlir::ConstantOp>::JLIRToStdConversionPattern;

    LogicalResult matchAndRewrite(jlir::ConstantOp op,
                                  ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        JuliaType type = op.getType().cast<JuliaType>();
        jl_datatype_t *julia_type = type.getDatatype();
        auto result = this->typeConverter->convertType(type);
        if (!result)
            return failure();

        if (jl_is_primitivetype(julia_type)) {
            int nb = jl_datatype_size(julia_type);
            APInt val((julia_type == jl_bool_type) ? 1 : (8 * nb), 0);
            void *bits = const_cast<uint64_t*>(val.getRawData());
            assert(llvm::sys::IsLittleEndianHost);
            memcpy(bits, op.value(), nb);

            Attribute value_attribute;
            if (FloatType ft = result.dyn_cast<FloatType>()) {
                APFloat fval(ft.getFloatSemantics(), val);
                value_attribute = rewriter.getFloatAttr(ft, fval);
            } else {
                value_attribute = rewriter.getIntegerAttr(result, val);
            }

            mlir::ConstantOp target = rewriter.create<mlir::ConstantOp>(op.getLoc(), value_attribute);
            rewriter.replaceOpWithNewOp<ConvertStdOp>(op, type, target.getResult());
            return success();
        }

        return failure();
    }
};

struct GotoOpLowering : public JLIRToStdConversionPattern<GotoOp> {
    using JLIRToStdConversionPattern<GotoOp>::JLIRToStdConversionPattern;

    LogicalResult matchAndRewrite(GotoOp op,
                                  ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        
        SmallVector<Value, 4> newOperands(operands.size());
        this->convertOperands(rewriter, op.getLoc(), operands, newOperands);

        rewriter.replaceOpWithNewOp<BranchOp>(
            op, op.getSuccessor(), newOperands);
        return success();
    }
};

struct GotoIfNotOpLowering : public JLIRToStdConversionPattern<GotoIfNotOp> {
    using JLIRToStdConversionPattern<GotoIfNotOp>::JLIRToStdConversionPattern;

    LogicalResult matchAndRewrite(GotoIfNotOp op,
                                  ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        unsigned nBranchOperands = op.branchOperands().size();
        unsigned nFallthroughOperands = op.fallthroughOperands().size();
        assert(operands.size() == nBranchOperands + nFallthroughOperands + 1);

        Value cond = this->convertValue(rewriter, op.getLoc(), operands.front());
        // TODO: Go from i8 to i1
        ValueRange branchOperands = operands.slice(1, nBranchOperands); 
        ValueRange fallthroughOperands = operands.slice(1 + nBranchOperands, nFallthroughOperands);

        SmallVector<Value, 4> newBranchOperands(branchOperands.size());
        this->convertOperands(rewriter, op.getLoc(), branchOperands, newBranchOperands);

        SmallVector<Value, 4> newFallthroughOperands(fallthroughOperands.size());
        this->convertOperands(rewriter, op.getLoc(), fallthroughOperands, newFallthroughOperands);

        rewriter.replaceOpWithNewOp<CondBranchOp>(op, cond,
                                                  op.fallthroughDest(), newFallthroughOperands,
                                                  op.branchDest(),      newBranchOperands);
        return success();
    }
};

struct ReturnOpLowering : public JLIRToStdConversionPattern<jlir::ReturnOp> {
    using JLIRToStdConversionPattern<jlir::ReturnOp>::JLIRToStdConversionPattern;

    LogicalResult matchAndRewrite(jlir::ReturnOp op,
                                  ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {

        SmallVector<Value, 2> newOperands(operands.size());
        this->convertOperands(rewriter, op.getLoc(), operands, newOperands);
        rewriter.replaceOpWithNewOp<mlir::ReturnOp>(op, newOperands);
        return success();
    }
};

struct NotIntOpLowering : public JLIRToStdConversionPattern<Intrinsic_not_int> {
    using JLIRToStdConversionPattern<Intrinsic_not_int>::JLIRToStdConversionPattern;

    LogicalResult matchAndRewrite(Intrinsic_not_int op,
                                  ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        
        SmallVector<Value, 2> newOperands(operands.size());
        this->convertOperands(rewriter, op.getLoc(), operands, newOperands);
        IntegerType type = newOperands.front().getType().cast<IntegerType>();

        mlir::ConstantOp maskConstantOp =
            rewriter.create<mlir::ConstantOp>(
                op.getLoc(), type,
                rewriter.getIntegerAttr(type,
                                        // need APInt for sign extension
                                        APInt(type.getWidth(), -1,
                                              /*isSigned=*/true)));

        XOrOp target = rewriter.create<XOrOp>(
            op.getLoc(), type, newOperands.front(), maskConstantOp.getResult());
        rewriter.replaceOpWithNewOp<ConvertStdOp>(op, type, target.getResult());
        return success();
    }
};

struct IsOpLowering : public JLIRToStdConversionPattern<Builtin_is> {
    using JLIRToStdConversionPattern<Builtin_is>::JLIRToStdConversionPattern;

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

struct ArrayrefOpLowering : public JLIRToStdConversionPattern<Builtin_arrayref> {
    using JLIRToStdConversionPattern<Builtin_arrayref>::JLIRToStdConversionPattern;

    LogicalResult matchAndRewrite(Builtin_arrayref op,
                                  ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        // TODO: boundschecking
        // arrayref(bool, array, indices...)
        Optional<Value> memref = convertArray(rewriter, op.getLoc(), operands[1]);

        if (!memref.hasValue())
            return failure();
        
        MemRefType memrefType = memref.getValue().getType().cast<MemRefType>();
        
        // indices are reversed because Julia is column-major, but MLIR is
        // row-major
        SmallVector<Value, 2> indices;
        int64_t rank = memrefType.getRank();
        if (rank > 1 && op.getNumOperands() == 3) {
            // linear index, take advantage of lack of bounds checking
            indices.assign(
                rank, getIndexConstant(rewriter, op.getLoc(), 0));
            indices[rank-1] = decrementIndex(rewriter, op.getLoc(), operands[2]);
        } else {
            indices.assign(rank, nullptr);
            assert(rank == op.getNumOperands() - 2);
            for (unsigned i = 0; i < rank; i++) {
                indices[rank-i-1] = decrementIndex(
                    rewriter, op.getLoc(), operands[i+2]);
            }
        }

        LoadOp loadOp = rewriter.create<LoadOp>(op.getLoc(), memrefType.getElementType(), memref.getValue(), indices);
        rewriter.replaceOpWithNewOp<ConvertStdOp>(op, op.getResult().getType(), loadOp.getResult());
        return success();
    }
};

struct ArraysetOpLowering : public JLIRToStdConversionPattern<Builtin_arrayset> {
    using JLIRToStdConversionPattern<Builtin_arrayset>::JLIRToStdConversionPattern;

    LogicalResult matchAndRewrite(Builtin_arrayset op,
                                  ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        // TODO: boundschecking
        // arrayset(bool, array, val, indices...)
        Optional<Value> memref = convertArray(rewriter, op.getLoc(), operands[1]);
        Value val = operands[2];

        if (!memref.hasValue())
            return failure();
        
        MemRefType memrefType = memref.getValue().getType().cast<MemRefType>();
        
        // indices are reversed because Julia is column-major, but MLIR is
        // row-major
        SmallVector<Value, 2> indices;
        int64_t rank = memrefType.getRank();
        if (rank > 1 && op.getNumOperands() == 4) {
            // linear index, take advantage of lack of bounds checking
            indices.assign(
                rank, getIndexConstant(rewriter, op.getLoc(), 0));
            indices[rank-1] = decrementIndex(rewriter, op.getLoc(), operands[3]);
        } else {
            indices.assign(rank, nullptr);
            assert(rank == op.getNumOperands() - 3);
            for (unsigned i = 0; i < rank; i++) {
                indices[rank-i-1] = decrementIndex(
                    rewriter, op.getLoc(), operands[i+3]);
            }
        }

        StoreOp storeOp = rewriter.create<StoreOp>(op.getLoc(), val, memref.getValue(), indices);
        rewriter.replaceOp(op, operands[1]);
        return success();
    }
};

struct ArraysizeOpLowering : public JLIRToStdConversionPattern<Builtin_arraysize> {
    using JLIRToStdConversionPattern<Builtin_arraysize>::JLIRToStdConversionPattern;

    LogicalResult matchAndRewrite(Builtin_arraysize op,
                                  ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        // TODO: Boundschecking
        // arraysize(array, ndim)
        Optional<Value> memref = convertArray(rewriter, op.getLoc(), operands[0]);

        if (!memref.hasValue())
            return failure();

        MemRefType memrefType = memref.getValue().getType().cast<MemRefType>();
        
        // indices are reversed because Julia is column-major, but MLIR is
        // row-major
        auto rank = getIndexConstant(rewriter, op.getLoc(), memrefType.getRank());

        Type indexType = rewriter.getIndexType();

        ConvertStdOp index = rewriter.create<ConvertStdOp>(op.getLoc(), indexType, operands[1]);
        SubIOp subOp = rewriter.create<SubIOp>(op.getLoc(), indexType, rank, index);

        DimOp dimOp = rewriter.create<DimOp>(op.getLoc(), memref.getValue(), subOp.getResult());
        rewriter.replaceOpWithNewOp<ConvertStdOp>(op, op.getResult().getType(), dimOp.getResult());
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

    // Only partial lowering occurs at this stage.
    target.addLegalOp<Builtin_is>();

    populateFuncOpTypeConversionPattern(patterns, &getContext(), converter);
    patterns.insert<
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
        ArrayrefOpLowering, // Builtin_arrayref
        // Builtin_const_arrayref
        ArraysetOpLowering, // Builtin_arrayset
        ArraysizeOpLowering, // Builtin_arraysize
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
                    getFunction(), target, patterns)))
        signalPassFailure();

}

std::unique_ptr<Pass> mlir::jlir::createJLIRToStandardLoweringPass() {
    return std::make_unique<JLIRToStandardLoweringPass>();
}
