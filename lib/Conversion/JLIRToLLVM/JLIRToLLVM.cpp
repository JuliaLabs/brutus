#include "brutus/Conversion/JLIRToLLVM/JLIRToLLVM.h"
#include "brutus/Conversion/JLIRToStandard/JLIRToStandard.h"

#include "juliapriv/julia_private.h"

#include "mlir/IR/Types.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"

#include "llvm/Support/SwapByteOrder.h"

using namespace mlir;
using namespace jlir;

JLIRToLLVMTypeConverter::JLIRToLLVMTypeConverter(MLIRContext *ctx, LowerToLLVMOptions options)
    : LLVMTypeConverter(ctx, options),
      llvmDialect(ctx->getOrLoadDialect<LLVM::LLVMDialect>()),
      voidType(LLVM::LLVMVoidType::get(ctx)),
      int1Type(IntegerType::get(ctx, 1)),
      int8Type(IntegerType::get(ctx, 8)),
      int16Type(IntegerType::get(ctx, 16)),
      int32Type(IntegerType::get(ctx, 32)),
      int64Type(IntegerType::get(ctx, 64)),
      sizeType((sizeof(size_t) == 8) ? int64Type : int32Type),
      longType((sizeof(long) == 8) ? int64Type : int32Type),
      mlirLongType((sizeof(long) == 8) ? IntegerType::get(ctx, 64) : IntegerType::get(ctx, 32)),
      jlvalueType(LLVM::LLVMStructType::getIdentified(ctx, "struct_jl_value_type")), // TODO: Add name
      pjlvalueType(LLVM::LLVMPointerType::get(jlvalueType)),
      jlarrayType(
          LLVM::LLVMStructType::getLiteral(
              ctx,
              ArrayRef<mlir::Type>({
                  LLVM::LLVMPointerType::get(int8Type), // data
                  sizeType,                             // length
                  int16Type,                            // flags
                  int16Type,                            // elsize
                  int32Type,                            // offset
                  sizeType                              // nrows
              }))),
      pjlarrayType(LLVM::LLVMPointerType::get(jlarrayType))
{

    static_assert(sizeof(jl_array_flags_t) == sizeof(int16_t), "sizeof(jl_array_flags) != sizeof(int16_t)");

    assert(llvmDialect && "LLVM IR dialect is not registered");
    addConversion([&](JuliaType jt) {
        return julia_type_to_llvm((jl_value_t *)jt.getDatatype());
    });

    // TESTING
    voidType = pjlvalueType;
}

Type JLIRToLLVMTypeConverter::julia_bitstype_to_llvm(jl_value_t *bt)
{
    assert(jl_is_primitivetype(bt));
    // TODO: jl_bool_type is actually i8, but llvm expects i1
    if (bt == (jl_value_t *)jl_bool_type)
        return int1Type;
    if (bt == (jl_value_t *)jl_int32_type)
        return int32Type;
    if (bt == (jl_value_t *)jl_int64_type)
        return int64Type;
    // if (llvmcall && (bt == (jl_value_t*)jl_float16_type))
    //     return Type::getHalfTy(llvmDialect);
    if (bt == (jl_value_t *)jl_float32_type)
        return FloatType::getF32(&getContext());
    if (bt == (jl_value_t *)jl_float64_type)
        return FloatType::getF64(&getContext());
    int nb = jl_datatype_size(bt);
    return IntegerType::get(&getContext(), nb * 8);
}

Type JLIRToLLVMTypeConverter::julia_struct_to_llvm(jl_value_t *jt)
{
    // this function converts a Julia Type into the equivalent LLVM struct
    // use this where C-compatible (unboxed) structs are desired
    // use julia_type_to_llvm directly when you want to preserve Julia's
    // type semantics
    if (jt == (jl_value_t *)jl_bottom_type)
        return voidType;
    if (jl_is_primitivetype(jt))
        return julia_bitstype_to_llvm(jt);
    jl_datatype_t *jst = (jl_datatype_t *)jt;
    if (jl_is_structtype(jt) && !(jst->layout && jl_is_layout_opaque(jst->layout)))
    {
        // bool is_tuple = jl_is_tuple_type(jt);
        jl_svec_t *ftypes = jl_get_fieldtypes(jst);
        size_t ntypes = jl_svec_len(ftypes);
        if (ntypes == 0 || (jst->layout && jl_datatype_nbits(jst) == 0))
            return voidType;

        // TODO: actually handle structs
    }

    return pjlvalueType; // prjlvalue?
}

Type JLIRToLLVMTypeConverter::julia_type_to_llvm(jl_value_t *jt)
{
    if (jt == jl_bottom_type || jt == (jl_value_t *)jl_void_type)
        return voidType;
    if (jl_is_concrete_immutable(jt))
    {
        if (jl_datatype_nbits(jt) == 0)
            return voidType;
        return julia_struct_to_llvm(jt);
    }

    return pjlvalueType; // prjlvalue?
}

// convert an LLVM type to same-sized int type
Type JLIRToLLVMTypeConverter::INTT(Type t)
{
    if (t.isInteger(32) || t.isInteger(64))
    {
        return t;
    }
    else if (LLVM::LLVMPointerType::isValidElementType(t))
    {
        if (sizeof(size_t) == 8)
        {
            return int64Type;
        }
        else
        {
            return int32Type;
        }
    }
    else if (t.isF64())
    {
        return int64Type;
    }
    else if (t.isF32())
    {
        return int32Type;
    }
    else if (t.isF16())
    {
        return int16Type;
    }

    unsigned nbits = LLVM::getPrimitiveTypeSizeInBits(t);
    assert(t != voidType && nbits > 0);
    return IntegerType::get(&getContext(), nbits);
}

namespace
{

    template <typename SourceOp>
    struct OpAndTypeConversionPattern : OpConversionPattern<SourceOp>
    {
        JLIRToLLVMTypeConverter &lowering;

        OpAndTypeConversionPattern(MLIRContext *ctx,
                                   JLIRToLLVMTypeConverter &lowering)
            : OpConversionPattern<SourceOp>(ctx), lowering(lowering) {}

        Value compareBits(ConversionPatternRewriter &rewriter, Location loc,
                          Value a, Value b) const
        {
            // TODO: check if this actually works--it may not, because new operands
            //       are being passed in through `a` and `b`, and they might not
            //       have a type yet

            assert(a.getType() == b.getType());
            Type t = a.getType().dyn_cast<Type>();

            if (t.isInteger(32) ||
                t.isInteger(64) ||
                LLVM::LLVMPointerType::isValidElementType(t) ||
                t.isF32())
            {

                Type t_int = lowering.INTT(t);
                if (t != t_int)
                {
                    a = rewriter.create<LLVM::BitcastOp>(loc, t_int, a).getResult();
                    b = rewriter.create<LLVM::BitcastOp>(loc, t_int, b).getResult();
                }
                return rewriter.create<LLVM::ICmpOp>(
                                   loc, LLVM::ICmpPredicate::eq, a, b)
                    .getResult();
            }

            // TODO
            assert(false && "unimplemented");
        }

        Value emitPointer(ConversionPatternRewriter &rewriter, Location loc,
                          jl_value_t *val) const
        {
            LLVM::ConstantOp addressOp = rewriter.create<LLVM::ConstantOp>(
                loc,
                lowering.longType,
                rewriter.getIntegerAttr(lowering.mlirLongType, (int64_t)val));
            return rewriter.create<LLVM::IntToPtrOp>(
                               loc, lowering.pjlvalueType, addressOp.getResult())
                .getResult();
        }

        // bitcast a `jl_value_t*` to a `jl_array_t*`
        Value emitPointerToArray(ConversionPatternRewriter &rewriter,
                                 Location loc,
                                 Value value) const
        {
            return rewriter.create<LLVM::BitcastOp>(
                               loc, lowering.pjlarrayType, value)
                .getResult();
        }

        // emit a pointer to the `nrows` field of a `jl_array_t`, given a `jl_value_t*`
        Value emitPointerToArrayField(ConversionPatternRewriter &rewriter,
                                      Location loc,
                                      Value pointerToArray,
                                      unsigned field) const
        {
            return rewriter.create<LLVM::GEPOp>(
                               loc,
                               LLVM::LLVMPointerType::get(lowering.sizeType),
                               pointerToArray,
                               llvm::makeArrayRef({// dereference `jl_array_t*` to get to `jl_array_t`
                                                   rewriter.create<LLVM::ConstantOp>(
                                                               loc,
                                                               lowering.int64Type,
                                                               rewriter.getI64IntegerAttr(0))
                                                       .getResult(),

                                                   // get `nrows` field
                                                   rewriter.create<LLVM::ConstantOp>(
                                                               loc,
                                                               lowering.int32Type,
                                                               rewriter.getI32IntegerAttr(field))
                                                       .getResult()}))
                .getResult();
        }

        // NOTE: `dimension` is 0-indexed
        Value emitArraySize(ConversionPatternRewriter &rewriter,
                            Location loc,
                            Value pointerToNrows,
                            Value dimension) const
        {
            Value pointerToSize = rewriter.create<LLVM::GEPOp>(
                loc,
                LLVM::LLVMPointerType::get(lowering.sizeType),
                pointerToNrows,
                llvm::makeArrayRef(dimension));
            return rewriter.create<LLVM::LoadOp>(
                               loc, lowering.sizeType, pointerToSize)
                .getResult();
        }
    };

    // is there some template magic that would allow us to combine
    // `ToLLVMOpPattern`, `ToUnaryLLVMOpPattern`, and `ToTernaryLLVMOpPattern`

    template <typename SourceOp, typename LLVMOp>
    struct ToLLVMOpPattern : public OpAndTypeConversionPattern<SourceOp>
    {
        using OpAndTypeConversionPattern<SourceOp>::OpAndTypeConversionPattern;

        LogicalResult matchAndRewrite(SourceOp op,
                                      ArrayRef<Value> operands,
                                      ConversionPatternRewriter &rewriter) const override
        {
            static_assert(
                std::is_base_of<OpTrait::OneResult<SourceOp>, SourceOp>::value,
                "expected single result op");
            rewriter.replaceOpWithNewOp<LLVMOp>(
                op, this->lowering.convertType(op.getType()), operands);
            return success();
        }
    };

    template <typename SourceOp, typename LLVMOp>
    struct ToUnaryLLVMOpPattern : public OpAndTypeConversionPattern<SourceOp>
    {
        using OpAndTypeConversionPattern<SourceOp>::OpAndTypeConversionPattern;

        LogicalResult matchAndRewrite(SourceOp op,
                                      ArrayRef<Value> operands,
                                      ConversionPatternRewriter &rewriter) const override
        {
            static_assert(
                std::is_base_of<OpTrait::OneResult<SourceOp>, SourceOp>::value,
                "expected single result op");
            assert(operands.size() == 1 && "expected unary operation");
            rewriter.replaceOpWithNewOp<LLVMOp>(
                op, this->lowering.convertType(op.getType()), operands.front());
            return success();
        }
    };

    template <typename SourceOp, typename LLVMOp>
    struct ToTernaryLLVMOpPattern : public OpAndTypeConversionPattern<SourceOp>
    {
        using OpAndTypeConversionPattern<SourceOp>::OpAndTypeConversionPattern;

        LogicalResult matchAndRewrite(SourceOp op,
                                      ArrayRef<Value> operands,
                                      ConversionPatternRewriter &rewriter) const override
        {
            static_assert(
                std::is_base_of<OpTrait::OneResult<SourceOp>, SourceOp>::value,
                "expected single result op");
            assert(operands.size() == 3 && "expected ternary operation");
            rewriter.replaceOpWithNewOp<LLVMOp>(
                op, this->lowering.convertType(op.getType()),
                operands[0],
                operands[1],
                operands[2]);
            return success();
        }
    };

    template <typename SourceOp>
    struct ToUndefOpPattern : public OpAndTypeConversionPattern<SourceOp>
    {
        using OpAndTypeConversionPattern<SourceOp>::OpAndTypeConversionPattern;

        LogicalResult matchAndRewrite(SourceOp op,
                                      ArrayRef<Value> operands,
                                      ConversionPatternRewriter &rewriter) const override
        {
            static_assert(
                std::is_base_of<OpTrait::OneResult<SourceOp>, SourceOp>::value,
                "expected single result op");
            rewriter.replaceOpWithNewOp<LLVM::UndefOp>(
                op, this->lowering.convertType(op.getType()));
            return success();
        }
    };

    struct ConvertStdOpLowering : public OpAndTypeConversionPattern<ConvertStdOp>
    {
        using OpAndTypeConversionPattern<ConvertStdOp>::OpAndTypeConversionPattern;

        LogicalResult
        matchAndRewrite(ConvertStdOp op,
                        ArrayRef<Value> operands,
                        ConversionPatternRewriter &rewriter) const override
        {
            // TODO: check that this conversion is valid
            rewriter.replaceOp(op, operands.front());
            return success();
        }
    };

    struct ConstantOpLowering : public OpAndTypeConversionPattern<ConstantOp>
    {
        using OpAndTypeConversionPattern<ConstantOp>::OpAndTypeConversionPattern;

        LogicalResult matchAndRewrite(ConstantOp op,
                                      ArrayRef<Value> operands,
                                      ConversionPatternRewriter &rewriter) const override
        {
            jl_value_t *value = op.value();
            jl_datatype_t *julia_type = op.getType().cast<JuliaType>().getDatatype();
            Type llvm_type = lowering.convertType(op.getType()).cast<Type>();

            if (llvm_type == lowering.voidType)
            {
                rewriter.replaceOp(
                    op, emitPointer(rewriter, op.getLoc(), op.value()));
                return success();
            }
            else if (jl_is_primitivetype(julia_type))
            {
                int nb = jl_datatype_size(julia_type);
                APInt val(8 * nb, 0);
                void *bits = const_cast<uint64_t *>(val.getRawData());
                assert(llvm::sys::IsLittleEndianHost);
                memcpy(bits, value, nb);

                Attribute value_attribute;
                if (llvm_type.isF32())
                {
                    APFloat fval(llvm_type.cast<FloatType>().getFloatSemantics(), val);
                    if (julia_type == jl_float32_type)
                    {
                        value_attribute = rewriter.getFloatAttr(
                            rewriter.getF32Type(), fval);
                    }
                    else if (julia_type == jl_float64_type)
                    {
                        value_attribute = rewriter.getFloatAttr(
                            rewriter.getF64Type(), fval);
                    }
                    else
                    {
                        assert(false && "not implemented");
                    }
                }
                else
                {
                    value_attribute = rewriter.getIntegerAttr(
                        rewriter.getIntegerType(nb * 8), val);
                }

                rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
                    op, llvm_type, value_attribute);
                return success();
            }
            else if (jl_is_structtype(julia_type))
            {
                // TODO
            }

            if (llvm_type == lowering.pjlvalueType)
            {
                rewriter.replaceOp(
                    op, emitPointer(rewriter, op.getLoc(), op.value()));
                return success();
            }

            rewriter.replaceOpWithNewOp<LLVM::UndefOp>(
                op, lowering.convertType(op.getType()));
            return success();
        }
    };

    struct GotoIfNotOpLowering : public OpAndTypeConversionPattern<GotoIfNotOp>
    {
        using OpAndTypeConversionPattern<GotoIfNotOp>::OpAndTypeConversionPattern;

        LogicalResult matchAndRewrite(GotoIfNotOp op,
                                      ArrayRef<Value> operands,
                                      ConversionPatternRewriter &rewriter) const override
        {
            assert(operands.size() >= 1);
            rewriter.replaceOpWithNewOp<LLVM::CondBrOp>(
                op, operands, op.getSuccessors(), op->getAttrs());
            return success();
        }
    };

    struct GotoOpLowering : public OpAndTypeConversionPattern<GotoOp>
    {
        using OpAndTypeConversionPattern<GotoOp>::OpAndTypeConversionPattern;

        LogicalResult matchAndRewrite(GotoOp op,
                                      ArrayRef<Value> operands,
                                      ConversionPatternRewriter &rewriter) const override
        {
            rewriter.replaceOpWithNewOp<LLVM::BrOp>(
                op, operands, op.getSuccessor(), op->getAttrs());
            return success();
        }
    };

    struct ReturnOpLowering : public OpAndTypeConversionPattern<ReturnOp>
    {
        using OpAndTypeConversionPattern<ReturnOp>::OpAndTypeConversionPattern;

        LogicalResult matchAndRewrite(ReturnOp op,
                                      ArrayRef<Value> operands,
                                      ConversionPatternRewriter &rewriter) const override
        {
            rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, operands);
            return success();
        }
    };

    struct NotIntOpLowering : public OpAndTypeConversionPattern<Intrinsic_not_int>
    {
        using OpAndTypeConversionPattern<Intrinsic_not_int>::OpAndTypeConversionPattern;

        LogicalResult matchAndRewrite(Intrinsic_not_int op,
                                      ArrayRef<Value> operands,
                                      ConversionPatternRewriter &rewriter) const override
        {
            jl_datatype_t *operand_type =
                op.getOperand(0).getType().dyn_cast<JuliaType>().getDatatype();
            bool is_bool = operand_type == jl_bool_type;
            uint64_t mask_value = is_bool ? 1 : -1;
            unsigned num_bits = 8 * (is_bool ? 1 : jl_datatype_size(operand_type));

            LLVM::ConstantOp mask_constant =
                rewriter.create<LLVM::ConstantOp>(
                    op.getLoc(), operands.front().getType(),
                    rewriter.getIntegerAttr(rewriter.getIntegerType(num_bits),
                                            // need APInt to do sign extension of mask
                                            APInt(num_bits, mask_value,
                                                  /*isSigned=*/true)));

            rewriter.replaceOpWithNewOp<LLVM::XOrOp>(
                op, operands.front().getType(),
                operands.front(), mask_constant.getResult());

            return success();
        }
    };

    struct IsOpLowering : public OpAndTypeConversionPattern<Builtin_is>
    {
        using OpAndTypeConversionPattern<Builtin_is>::OpAndTypeConversionPattern;

        LogicalResult matchAndRewrite(Builtin_is op,
                                      ArrayRef<Value> operands,
                                      ConversionPatternRewriter &rewriter) const override
        {
            assert(operands.size() == 2);
            jl_value_t *jt1 = (jl_value_t *)op.getOperand(0).getType().cast<JuliaType>().getDatatype();
            jl_value_t *jt2 = (jl_value_t *)op.getOperand(1).getType().cast<JuliaType>().getDatatype();
            if (jl_is_concrete_type(jt1) && jl_is_concrete_type(jt2) && !jl_is_kind(jt1) && !jl_is_kind(jt2) && jt1 != jt2)
            {
                // disjoint concrete leaf types are never equal
                assert(false && "should have been handled in JLIRToStandard");
            }

            // TODO: ghosts, see `emit_f_is`

            if (jl_type_intersection(jt1, jt2) == (jl_value_t *)jl_bottom_type)
            {
                assert(false && "should have been handled in JLIRToStandard");
            }

            bool justbits1 = jl_is_concrete_immutable(jt1);
            bool justbits2 = jl_is_concrete_immutable(jt2);
            if (justbits1 || justbits2)
            {
                if (jt1 == jt2)
                {
                    rewriter.replaceOp(
                        op, compareBits(
                                rewriter, op.getLoc(), operands[0], operands[1]));
                    return success();
                }

                // TODO
            }

            // TODO: `emit_box_compare`

            return failure();
        }
    };
} // namespace

// TODO: maybe values that convert to void should not be removed--pass a
//       pointer?
//
//       f(x::Bool) = x ? nothing : 100

void JLIRToLLVMLoweringPass::runOnFunction()
{
    RewritePatternSet patterns(&getContext());

    LowerToLLVMOptions options(&getContext());
    options.allocLowering = LowerToLLVMOptions::AllocLowering::AlignedAlloc;
    
    JLIRToLLVMTypeConverter converter(&getContext(), options);

    patterns.insert<
        ConvertStdOpLowering,
        ToUndefOpPattern<UnimplementedOp>,
        ToUndefOpPattern<UndefOp>,
        ConstantOpLowering, // (also JLIRTOStandard)
        ToUndefOpPattern<CallOp>,
        ToUndefOpPattern<InvokeOp>,
        GotoOpLowering,
        GotoIfNotOpLowering,
        ReturnOpLowering,
        ToUndefOpPattern<PiOp>,
        // Intrinsic_bitcast
        // Intrinsic_neg_int
        // Intrinsic_add_int  (JLIRToStandard)
        // Intrinsic_sub_int  (JLIRToStandard)
        // Intrinsic_mul_int  (JLIRToStandard)
        // Intrinsic_sdiv_int (JLIRToStandard)
        // Intrinsic_udiv_int (JLIRToStandard)
        // Intrinsic_srem_int (JLIRToStandard)
        // Intrinsic_urem_int (JLIRToStandard)
        // Intrinsic_add_ptr
        // Intrinsic_sub_ptr
        // Intrinsic_neg_float (JLIRToStandard)
        // Intrinsic_add_float (JLIRToStandard)
        // Intrinsic_sub_float (JLIRToStandard)
        // Intrinsic_mul_float (JLIRToStandard)
        // Intrinsic_div_float (JLIRToStandard)
        // Intrinsic_rem_float (JLIRToStandard)
        ToTernaryLLVMOpPattern<Intrinsic_fma_float, LLVM::FMAOp>,
        // Intrinsic_muladd_float
        // Intrinsic_neg_float_fast
        // Intrinsic_add_float_fast
        // Intrinsic_sub_float_fast
        // Intrinsic_mul_float_fast
        // Intrinsic_div_float_fast
        // Intrinsic_rem_float_fast
        // Intrinsic_eq_int   (JLIRToStandard)
        // Intrinsic_ne_int   (JLIRToStandard)
        // Intrinsic_slt_int  (JLIRToStandard)
        // Intrinsic_ult_int  (JLIRToStandard)
        // Intrinsic_sle_int  (JLIRToStandard)
        // Intrinsic_ule_int  (JLIRToStandard)
        // Intrinsic_eq_float (JLIRToStandard)
        // Intrinsic_ne_float (JLIRToStandard)
        // Intrinsic_lt_float (JLIRToStandard)
        // Intrinsic_le_float (JLIRToStandard)
        // Intrinsic_fpiseq
        // Intrinsic_fpislt
        // Intrinsic_and_int  (JLIRToStandard)
        // Intrinsic_or_int   (JLIRToStandard)
        // Intrinsic_xor_int  (JLIRToStandard)
        // Intrinsic_not_int  (JLIRToStandard)
        // Intrinsic_shl_int  (JLIRToStandard)
        // Intrinsic_lshr_int (JLIRToStandard)
        // Intrinsic_ashr_int (JLIRToStandard)
        // Intrinsic_bswap_int
        // Intrinsic_ctpop_int
        // Intrinsic_ctlz_int
        // Intrinsic_cttz_int
        // Intrinsic_sext_int  (JLIRToStandard)
        // Intrinsic_zext_int  (JLIRToStandard)
        // Intrinsic_trunc_int (JLIRToStandard)
        // Intrinsic_fptoui
        // Intrinsic_fptosi
        // Intrinsic_uitofp
        // Intrinsic_sitofp  (JLIRToStandard)
        // Intrinsic_fptrunc (JLIRToStandard)
        // Intrinsic_fpext   (JLIRToStandard)
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
        // Intrinsic_abs_float      (JLIRToStandard)
        // Intrinsic_copysign_float (JLIRToStandard)
        // Intrinsic_flipsign_int
        // Intrinsic_ceil_llvm (JLIRToStandard)
        // Intrinsic_floor_llvm
        ToUnaryLLVMOpPattern<Intrinsic_trunc_llvm, LLVM::TruncOp>,
        // Intrinsic_rint_llvm
        // Intrinsic_sqrt_llvm (JLIRToStandard)
        // Intrinsic_sqrt_llvm_fast
        // Intrinsic_pointerref
        // Intrinsic_pointerset
        // Intrinsic_cglobal
        // Intrinsic_llvmcall
        // Intrinsic_arraylen
        // Intrinsic_cglobal_auto
        // Builtin_throw
        IsOpLowering, // Builtin_is (also JLIRToStandard)
        // Builtin_typeof
        // Builtin_sizeof
        // Builtin_issubtype
        ToUndefOpPattern<Builtin_isa>, // Builtin_isa
        // Builtin__apply
        // Builtin__apply_pure
        // Builtin__apply_latest
        // Builtin__apply_iterate
        // Builtin_isdefined
        // Builtin_nfields
        // Builtin_tuple
        // Builtin_svec
        ToUndefOpPattern<Builtin_getfield>, // Builtin_getfield
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
        ToTernaryLLVMOpPattern<
            Builtin_ifelse, LLVM::SelectOp> // (also JLIRToStandard)
        // Builtin__typevar
        // invoke_kwsorter?
        >(&getContext(), converter);

    populateStdToLLVMConversionPatterns(converter, patterns);

    LLVMConversionTarget target(getContext());
    if (failed(applyPartialConversion(
            getFunction(), target, std::move(patterns))))
        signalPassFailure();
}

std::unique_ptr<Pass> mlir::jlir::createJLIRToLLVMLoweringPass()
{
    return std::make_unique<JLIRToLLVMLoweringPass>();
}
