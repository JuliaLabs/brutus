#include "brutus/Conversion/JLIRToStandard/JLIRToStandard.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Types.h"

#include "mlir/Dialect/StandardOps/Transforms/FuncConversions.h"

#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
using namespace mlir::jlir;

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

    addSourceMaterialization([&](OpBuilder &builder,
                                 Type resultType, ValueRange inputs,
                                 Location loc) -> Optional<Value> {
        if (inputs.size() == 1) {
            Value in = inputs.front();
            ConvertStdOp op = builder.create<ConvertStdOp>(loc, resultType, in);
            return op.getResult();
        } else {
            return None;
        }
    });

    addTargetMaterialization([&](OpBuilder &builder,
                                 Type resultType, ValueRange inputs,
                                 Location loc) -> Optional<Value> {
        if (inputs.size() == 1) {
            Value in = inputs.front();
            ConvertStdOp op = builder.create<ConvertStdOp>(loc, resultType, in);
            return op.getResult();
        } else {
            return None;
        }
    });

    addArgumentMaterialization([&](OpBuilder &builder,
                                   Type resultType, ValueRange inputs,
                                   Location loc) -> Optional<Value> {
        if (inputs.size() == 1) {
            Value in = inputs.front();
            ConvertStdOp op = builder.create<ConvertStdOp>(loc, resultType, in);
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
    } else if (jl_is_array_type(jdt)) {
        JuliaType elJLType = JuliaType::get(t.getContext(), (jl_datatype_t*)jl_tparam0(jdt));
        Optional<Type> elType = convertJuliaType(elJLType);
        if (!elType.hasValue()) {
            return None;
        }
        unsigned rank = jl_unbox_uint64(jl_tparam1(jdt));
        SmallVector<int64_t, 2> shape(rank, -1);
        return MemRefType::get(shape, elType.getValue());
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
        return IntegerType::get(ctx, 1);
    } else if (jdt == jl_int32_type)
        return IntegerType::get(ctx, 32);
    else if (jdt == jl_int64_type)
        return IntegerType::get(ctx, 64);
    else if (jdt == jl_float32_type)
        return FloatType::getF32(ctx);
    else if (jdt == jl_float64_type)
        return FloatType::getF64(ctx);
    int nb = jl_datatype_size(jdt);
    return IntegerType::get(ctx, nb * 8);
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

        rewriter.replaceOpWithNewOp<StdOp>(op, result, operands, op.getAttrs());

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


        rewriter.replaceOpWithNewOp<CmpOp>(op, predicate, operands[0], operands[1]);
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
        
        rewriter.replaceOpWithNewOp<BranchOp>(
            op, op.getSuccessor(), operands);
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

        rewriter.replaceOpWithNewOp<CondBranchOp>(op, cond,
                                                  op.fallthroughDest(), fallthroughOperands,
                                                  op.branchDest(),      branchOperands);
        return success();
    }
};

struct ReturnOpLowering : public JLIRToStdConversionPattern<jlir::ReturnOp> {
    using JLIRToStdConversionPattern<jlir::ReturnOp>::JLIRToStdConversionPattern;

    LogicalResult matchAndRewrite(jlir::ReturnOp op,
                                  ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {

        rewriter.replaceOpWithNewOp<mlir::ReturnOp>(op, operands);
        return success();
    }
};

struct NotIntOpLowering : public JLIRToStdConversionPattern<Intrinsic_not_int> {
    using JLIRToStdConversionPattern<Intrinsic_not_int>::JLIRToStdConversionPattern;

    LogicalResult matchAndRewrite(Intrinsic_not_int op,
                                  ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        
        IntegerType type = operands.front().getType().cast<IntegerType>();

        mlir::ConstantOp maskConstantOp =
            rewriter.create<mlir::ConstantOp>(
                op.getLoc(), type,
                rewriter.getIntegerAttr(type,
                                        // need APInt for sign extension
                                        APInt(type.getWidth(), -1,
                                              /*isSigned=*/true)));

        rewriter.replaceOpWithNewOp<XOrOp>(
            op, type, operands.front(), maskConstantOp.getResult());
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
        Value memref = operands[1];
        
        if (auto memrefType = memref.getType().dyn_cast<MemRefType>()) {
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

            rewriter.replaceOpWithNewOp<LoadOp>(op, memrefType.getElementType(), memref, indices);
            return success();
        }
        return failure();
    }
};

struct ArraysetOpLowering : public JLIRToStdConversionPattern<Builtin_arrayset> {
    using JLIRToStdConversionPattern<Builtin_arrayset>::JLIRToStdConversionPattern;

    LogicalResult matchAndRewrite(Builtin_arrayset op,
                                  ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        // arrayset(i1, Array, val, indices...)
        Value memref = operands[1];
        
        if (auto memrefType = memref.getType().dyn_cast<MemRefType>()) {
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

            rewriter.create<StoreOp>(op.getLoc(), operands[2], memref, indices);
            rewriter.replaceOp(op, operands[1]);
            return success();
        }
        return failure();
    }
};

struct ArraysizeOpLowering : public JLIRToStdConversionPattern<Builtin_arraysize> {
    using JLIRToStdConversionPattern<Builtin_arraysize>::JLIRToStdConversionPattern;

    LogicalResult matchAndRewrite(Builtin_arraysize op,
                                  ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        // TODO: Boundschecking
        // arraysize(array, ndim)
        Value memref = operands[0];
        
        if (auto memrefType = memref.getType().dyn_cast<MemRefType>()) {        
            // indices are reversed because Julia is column-major, but MLIR is
            // row-major
            auto rank = getIndexConstant(rewriter, op.getLoc(), memrefType.getRank());

            Type indexType = rewriter.getIndexType();

            ConvertStdOp index = rewriter.create<ConvertStdOp>(op.getLoc(), indexType, operands[1]);
            SubIOp subOp = rewriter.create<SubIOp>(op.getLoc(), indexType, rank, index);

            rewriter.replaceOpWithNewOp<DimOp>(op, memref, subOp.getResult());
            return success();
        }
        return failure();
    }
};

} // namespace

void mlir::jlir::populateJLIRToStdConversionPatterns(OwningRewritePatternList &patterns, 
                                         MLIRContext &context,
                                         JLIRToStandardTypeConverter &converter) {
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
        >(&context, converter);
}

namespace {
struct JLIRToStandardLoweringPass
    : public PassWrapper<JLIRToStandardLoweringPass, OperationPass<ModuleOp>> {

    static bool isFuncOpLegal(FuncOp op, JLIRToStandardTypeConverter &converter);
    static bool isCallOpLegal(mlir::CallOp op, JLIRToStandardTypeConverter &converter);
    void runOnOperation() override;
};
} // namespace

bool JLIRToStandardLoweringPass::isFuncOpLegal(
    FuncOp op, JLIRToStandardTypeConverter &converter) {

    FunctionType ft = op.getType().cast<FunctionType>();
    // function is illegal if any of its input or result types can but haven't
    // been converted

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

void JLIRToStandardLoweringPass::runOnOperation() {
    auto module = getOperation();
    ConversionTarget target(getContext());
    JLIRToStandardTypeConverter converter(&getContext());

    target.addLegalDialect<StandardOpsDialect>();
    target.addLegalOp<ConvertStdOp>();
    target.addLegalOp<UnimplementedOp>();
    // Only partial lowering occurs at this stage.
    target.addLegalOp<Builtin_is>();

    OwningRewritePatternList patterns;
    populateJLIRToStdConversionPatterns(patterns, getContext(), converter);

    target.addDynamicallyLegalOp<FuncOp>([this, &converter](FuncOp op) {
        return isFuncOpLegal(op, converter);
    });
    populateFuncOpTypeConversionPattern(patterns, &getContext(), converter);

    if (failed(applyPartialConversion(
                    module, target, patterns)))
        signalPassFailure();

}

std::unique_ptr<OperationPass<ModuleOp>> mlir::jlir::createJLIRToStandardLoweringPass() {
    return std::make_unique<JLIRToStandardLoweringPass>();
}
