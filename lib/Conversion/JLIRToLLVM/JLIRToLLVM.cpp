#include "brutus/Dialect/Julia/JuliaOps.h"
#include "brutus/Conversion/JLIRToLLVM/JLIRToLLVM.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/SwapByteOrder.h"

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
        // TODO: try this later
        //     [&](JuliaType jt, SmallVectorImpl<Type> &results) {
        //         LLVM::LLVMType converted =
        //             julia_type_to_llvm((jl_value_t*)jt.getDatatype());
        //         // drop value if it converts to void type
        //         if (converted != void_type) {
        //             results.push_back(converted);
        //         }
        //         return success(); });
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

    // convert an LLVM type to same-sized int type
    LLVM::LLVMType INTT(LLVM::LLVMType t) {
        if (t.isIntegerTy()) {
            return t;
        } else if (t.isPointerTy()) {
            if (sizeof(size_t) == 8) {
                return LLVM::LLVMType::getInt64Ty(llvm_dialect);
            } else {
                return LLVM::LLVMType::getInt32Ty(llvm_dialect);
            }
        } else if (t.isDoubleTy()) {
            return LLVM::LLVMType::getInt64Ty(llvm_dialect);
        } else if (t.isFloatTy()) {
            return LLVM::LLVMType::getInt32Ty(llvm_dialect);
        } else if (t.isHalfTy()) {
            return LLVM::LLVMType::getInt16Ty(llvm_dialect);
        }

        unsigned nbits = t.getUnderlyingType()->getPrimitiveSizeInBits();
        assert(t != void_type && nbits > 0);
        return LLVM::LLVMType::getIntNTy(llvm_dialect, nbits);
    }
};

namespace {

template <typename SourceOp>
struct OpAndTypeConversionPattern : OpConversionPattern<SourceOp> {
    JLIRToLLVMTypeConverter &lowering;

    OpAndTypeConversionPattern(MLIRContext *ctx,
                               JLIRToLLVMTypeConverter &lowering)
        : OpConversionPattern<SourceOp>(ctx), lowering(lowering) {}

    // truncate a Bool (i8) to an i1
    Value truncateBool(Location loc, Value b, ConversionPatternRewriter &rewriter) const {
        LLVM::TruncOp truncated = rewriter.create<LLVM::TruncOp>(
            loc, LLVM::LLVMType::getInt1Ty(lowering.llvm_dialect), b);
        return truncated.getResult();
    }

    // zero extend an i1 to a Bool (i8)
    Value extendBool(Location loc, Value b, ConversionPatternRewriter &rewriter) const {
        LLVM::ZExtOp extended = rewriter.create<LLVM::ZExtOp>(
            loc, LLVM::LLVMType::getInt8Ty(lowering.llvm_dialect), b);
        return extended.getResult();
    }

    Value compareBits(Location loc, Value a, Value b, ConversionPatternRewriter &rewriter) const {
        assert(a.getType() == b.getType());
        LLVM::LLVMType t = a.getType().dyn_cast<LLVM::LLVMType>();

        if (t.isIntegerTy() || t.isPointerTy()
            || t.getUnderlyingType()->isFloatingPointTy()) {

            LLVM::LLVMType t_int = lowering.INTT(t);
            if (t != t_int) {
                a = rewriter.create<LLVM::BitcastOp>(loc, t_int, a).getResult();
                b = rewriter.create<LLVM::BitcastOp>(loc, t_int, b).getResult();
            }
            Value result = rewriter.create<LLVM::ICmpOp>(
                loc, LLVM::ICmpPredicate::eq, a, b).getResult();
            // do I really need to extend back to i8?
            return extendBool(loc, result, rewriter);
        }

        // TODO
        assert(false && "unimplemented");
    }
};

// is there some template magic that would allow us to combine
// `ToLLVMOpPattern`, `ToUnaryLLVMOpPattern`, and `ToTernaryLLVMOpPattern`?

template <typename SourceOp, typename LLVMOp>
struct ToLLVMOpPattern : public OpAndTypeConversionPattern<SourceOp> {
    using OpAndTypeConversionPattern<SourceOp>::OpAndTypeConversionPattern;

    LogicalResult matchAndRewrite(SourceOp op,
                                  ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        static_assert(
            std::is_base_of<OpTrait::OneResult<SourceOp>, SourceOp>::value,
            "expected single result op");
        rewriter.replaceOpWithNewOp<LLVMOp>(
            op, this->lowering.convertToLLVMType(op.getType()), operands);
        return success();
    }
};

template <typename SourceOp, typename LLVMOp>
struct ToUnaryLLVMOpPattern : public OpAndTypeConversionPattern<SourceOp> {
    using OpAndTypeConversionPattern<SourceOp>::OpAndTypeConversionPattern;

    LogicalResult matchAndRewrite(SourceOp op,
                                  ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        static_assert(
            std::is_base_of<OpTrait::OneResult<SourceOp>, SourceOp>::value,
            "expected single result op");
        assert(operands.size() == 1 && "expected unary operation");
        rewriter.replaceOpWithNewOp<LLVMOp>(
            op, this->lowering.convertToLLVMType(op.getType()), operands.front());
        return success();
    }
};

template <typename SourceOp, typename LLVMOp>
struct ToTernaryLLVMOpPattern : public OpAndTypeConversionPattern<SourceOp> {
    using OpAndTypeConversionPattern<SourceOp>::OpAndTypeConversionPattern;

    LogicalResult matchAndRewrite(SourceOp op,
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
        return success();
    }
};

template <typename SourceOp>
struct ToUndefOpPattern : public OpAndTypeConversionPattern<SourceOp> {
    using OpAndTypeConversionPattern<SourceOp>::OpAndTypeConversionPattern;

    LogicalResult matchAndRewrite(SourceOp op,
                                  ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        static_assert(
            std::is_base_of<OpTrait::OneResult<SourceOp>, SourceOp>::value,
            "expected single result op");
        rewriter.replaceOpWithNewOp<LLVM::UndefOp>(
            op, this->lowering.convertToLLVMType(op.getType()));
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
        CmpOp cmp = rewriter.create<CmpOp>(
            op.getLoc(), predicate, operands[0], operands[1]);
        // assumes a Bool (i8) is to be returned
        rewriter.replaceOp(
            op, this->extendBool(op.getLoc(), cmp.getResult(), rewriter));
        return success();
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

    LogicalResult matchAndRewrite(FuncOp op,
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
        return success();
    }
};

struct ConstantOpLowering : public OpAndTypeConversionPattern<ConstantOp> {
    using OpAndTypeConversionPattern<ConstantOp>::OpAndTypeConversionPattern;

    LogicalResult matchAndRewrite(ConstantOp op,
                                  ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        jl_value_t *value = op.value();
        jl_datatype_t *julia_type = op.getType().cast<JuliaType>().getDatatype();
        LLVM::LLVMType llvm_type = lowering.convertToLLVMType(op.getType());

        if (llvm_type == lowering.void_type) {
            rewriter.replaceOpWithNewOp<LLVM::UndefOp>(op, llvm_type);
            return success();

        } else if (jl_is_primitivetype(julia_type)) {
            int nb = jl_datatype_size(julia_type);
            APInt val(8 * nb, 0);
            void *bits = const_cast<uint64_t*>(val.getRawData());
            assert(llvm::sys::IsLittleEndianHost);
            memcpy(bits, value, nb);

            Attribute value_attribute;
            llvm::Type *underlying_llvm_type = llvm_type.getUnderlyingType();
            if (underlying_llvm_type->isFloatingPointTy()) {
                APFloat fval(underlying_llvm_type->getFltSemantics(), val);
                if (julia_type == jl_float32_type) {
                    value_attribute = rewriter.getFloatAttr(
                        rewriter.getF32Type(), fval);
                } else if (julia_type == jl_float64_type) {
                    value_attribute = rewriter.getFloatAttr(
                        rewriter.getF64Type(), fval);
                } else {
                    assert(false && "not implemented");
                }
            } else {
                value_attribute = rewriter.getIntegerAttr(
                    rewriter.getIntegerType(nb*8), val);
            }

            rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
                op, llvm_type, value_attribute);
            return success();

        } else if (jl_is_structtype(julia_type)) {
            // TODO
        }

        if (llvm_type == lowering.pjlvalue) {
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
            return success();
        }

        rewriter.replaceOpWithNewOp<LLVM::UndefOp>(
            op, lowering.convertToLLVMType(op.getType()));
        return success();
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

    LogicalResult matchAndRewrite(GotoOp op,
                                  ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<LLVM::BrOp>(
            op, operands, op.getSuccessor());
        return success();
    }
};

struct GotoIfNotOpLowering : public OpAndTypeConversionPattern<GotoIfNotOp> {
    using OpAndTypeConversionPattern<GotoIfNotOp>::OpAndTypeConversionPattern;

    LogicalResult matchAndRewrite(GotoIfNotOp op,
                                  ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        assert(operands.size() >= 1);

        // truncate condition from i8 to i1
        SmallVector<Value, 4> new_operands;
        std::copy(operands.begin(), operands.end(),
                  std::back_inserter(new_operands));
        new_operands.front() = truncateBool(
            op.getLoc(), operands.front(), rewriter);

        rewriter.replaceOpWithNewOp<LLVM::CondBrOp>(
            op, new_operands, op.getSuccessors(), op.getAttrs());
        return success();
    }
};

struct ReturnOpLowering : public OpAndTypeConversionPattern<ReturnOp> {
    using OpAndTypeConversionPattern<ReturnOp>::OpAndTypeConversionPattern;

    LogicalResult matchAndRewrite(ReturnOp op,
                                  ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        // drop operand if its type is the LLVM void type
        if (operands.size() == 1
            && operands.front().getType() == lowering.void_type) {
            operands = llvm::None;
        }
        rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, operands);
        return success();
    }
};

struct PiOpLowering : public ToUndefOpPattern<PiOp> {
    // TODO
    using ToUndefOpPattern<PiOp>::ToUndefOpPattern;
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

struct IsOpLowering : public OpAndTypeConversionPattern<Builtin_is> {
    using OpAndTypeConversionPattern<Builtin_is>::OpAndTypeConversionPattern;

    LogicalResult matchAndRewrite(Builtin_is op,
                                  ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        // should result be an i1 or i8? `emit_f_is` uses i1 but Bool is i8

        assert(operands.size() == 2);
        jl_value_t *t1 = (jl_value_t*)op.getOperand(0).getType()
            .dyn_cast<JuliaType>().getDatatype();
        jl_value_t *t2 = (jl_value_t*)op.getOperand(1).getType()
            .dyn_cast<JuliaType>().getDatatype();
        if (jl_is_concrete_type(t1) && jl_is_concrete_type(t2)
            && !jl_is_kind(t1) && !jl_is_kind(t2) && t1 != t2) {
            // disjoint concrete leaf types are never equal
            rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
                op, LLVM::LLVMType::getInt8Ty(lowering.llvm_dialect),
                rewriter.getI8IntegerAttr(0));
            return success();
        }

        // TODO: ghosts, see `emit_f_is`

        if (jl_type_intersection(t1, t2) == (jl_value_t*)jl_bottom_type) {
            rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
                op, LLVM::LLVMType::getInt8Ty(lowering.llvm_dialect),
                rewriter.getI8IntegerAttr(0));
            return success();
        }

        bool justbits1 = jl_is_concrete_immutable(t1);
        bool justbits2 = jl_is_concrete_immutable(t2);
        if (justbits1 || justbits2) {
            if (t1 == t2) {
                rewriter.replaceOp(
                    op, compareBits(
                        op.getLoc(), operands[0], operands[1], rewriter));
                return success();
            }

            // TODO
        }

        // TODO: `emit_box_compare`

        return failure();
    }
};

struct IfElseOpLowering : public OpAndTypeConversionPattern<Builtin_ifelse> {
    using OpAndTypeConversionPattern<Builtin_ifelse>::OpAndTypeConversionPattern;

    LogicalResult matchAndRewrite(Builtin_ifelse op,
                                  ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const override {
        assert(operands.size() == 3);
        Value condition = truncateBool(op.getLoc(), operands.front(), rewriter);
        rewriter.replaceOpWithNewOp<LLVM::SelectOp>(
            op, condition, operands[1], operands[2]);
        return success();
    }
};

} // namespace

struct JLIRToLLVMLoweringPass : public PassWrapper<JLIRToLLVMLoweringPass, FunctionPass> {
    void runOnFunction() final {
        ConversionTarget target(getContext());
        target.addLegalDialect<LLVM::LLVMDialect>();

        OwningRewritePatternList patterns;
        JLIRToLLVMTypeConverter converter(&getContext());
        patterns.insert<
        //     FuncOpConversion,
            ToUndefOpPattern<UnimplementedOp>,
            ToUndefOpPattern<UndefOp>
        //     ConstantOpLowering,
        //     CallOpLowering,
        //     InvokeOpLowering,
        //     GotoIfNotOpLowering,
        //     ReturnOpLowering,
        //     PiOpLowering,
        //     // Intrinsic_bitcast
        //     // Intrinsic_neg_int
        //     ToLLVMOpPattern<Intrinsic_add_int, LLVM::AddOp>,
        //     ToLLVMOpPattern<Intrinsic_sub_int, LLVM::SubOp>,
        //     ToLLVMOpPattern<Intrinsic_mul_int, LLVM::MulOp>,
        //     ToLLVMOpPattern<Intrinsic_sdiv_int, LLVM::SDivOp>,
        //     ToLLVMOpPattern<Intrinsic_udiv_int, LLVM::UDivOp>,
        //     ToLLVMOpPattern<Intrinsic_srem_int, LLVM::SRemOp>,
        //     ToLLVMOpPattern<Intrinsic_urem_int, LLVM::URemOp>,
        //     // Intrinsic_add_ptr
        //     // Intrinsic_sub_ptr
        //     ToLLVMOpPattern<Intrinsic_neg_float, LLVM::FNegOp>,
        //     ToLLVMOpPattern<Intrinsic_add_float, LLVM::FAddOp>,
        //     ToLLVMOpPattern<Intrinsic_sub_float, LLVM::FSubOp>,
        //     ToLLVMOpPattern<Intrinsic_mul_float, LLVM::FMulOp>,
        //     ToLLVMOpPattern<Intrinsic_div_float, LLVM::FDivOp>,
        //     ToLLVMOpPattern<Intrinsic_rem_float, LLVM::FRemOp>,
        //     ToTernaryLLVMOpPattern<Intrinsic_fma_float, LLVM::FMAOp>,
        //     // Intrinsic_muladd_float
        //     // Intrinsic_neg_float_fast
        //     // Intrinsic_add_float_fast
        //     // Intrinsic_sub_float_fast
        //     // Intrinsic_mul_float_fast
        //     // Intrinsic_div_float_fast
        //     // Intrinsic_rem_float_fast
        //     ToICmpOpPattern<Intrinsic_eq_int, LLVM::ICmpPredicate::eq>,
        //     ToICmpOpPattern<Intrinsic_ne_int, LLVM::ICmpPredicate::ne>,
        //     ToICmpOpPattern<Intrinsic_slt_int, LLVM::ICmpPredicate::slt>,
        //     ToICmpOpPattern<Intrinsic_ult_int, LLVM::ICmpPredicate::ult>,
        //     ToICmpOpPattern<Intrinsic_sle_int, LLVM::ICmpPredicate::sle>,
        //     ToICmpOpPattern<Intrinsic_ule_int, LLVM::ICmpPredicate::ule>,
        //     ToFCmpOpPattern<Intrinsic_eq_float, LLVM::FCmpPredicate::oeq>,
        //     ToFCmpOpPattern<Intrinsic_ne_float, LLVM::FCmpPredicate::une>,
        //     ToFCmpOpPattern<Intrinsic_lt_float, LLVM::FCmpPredicate::olt>,
        //     ToFCmpOpPattern<Intrinsic_le_float, LLVM::FCmpPredicate::ole>,
        //     // Intrinsic_fpiseq
        //     // Intrinsic_fpislt
        //     ToLLVMOpPattern<Intrinsic_and_int, LLVM::AndOp>,
        //     ToLLVMOpPattern<Intrinsic_or_int, LLVM::OrOp>,
        //     ToLLVMOpPattern<Intrinsic_xor_int, LLVM::XOrOp>,
        //     NotIntOpLowering, // Intrinsic_not_int
        //     // Intrinsic_shl_int
        //     // Intrinsic_lshr_int
        //     // Intrinsic_ashr_int
        //     // Intrinsic_bswap_int
        //     // Intrinsic_ctpop_int
        //     // Intrinsic_ctlz_int
        //     // Intrinsic_cttz_int
        //     // Intrinsic_sext_int
        //     // Intrinsic_zext_int
        //     // Intrinsic_trunc_int
        //     // Intrinsic_fptoui
        //     // Intrinsic_fptosi
        //     // Intrinsic_uitofp
        //     // Intrinsic_sitofp
        //     // Intrinsic_fptrunc
        //     // Intrinsic_fpext
        //     // Intrinsic_checked_sadd_int
        //     // Intrinsic_checked_uadd_int
        //     // Intrinsic_checked_ssub_int
        //     // Intrinsic_checked_usub_int
        //     // Intrinsic_checked_smul_int
        //     // Intrinsic_checked_umul_int
        //     // Intrinsic_checked_sdiv_int
        //     // Intrinsic_checked_udiv_int
        //     // Intrinsic_checked_srem_int
        //     // Intrinsic_checked_urem_int
        //     ToUnaryLLVMOpPattern<Intrinsic_abs_float, LLVM::FAbsOp>,
        //     // Intrinsic_copysign_float
        //     // Intrinsic_flipsign_int
        //     ToUnaryLLVMOpPattern<Intrinsic_ceil_llvm, LLVM::FCeilOp>,
        //     // Intrinsic_floor_llvm
        //     ToUnaryLLVMOpPattern<Intrinsic_trunc_llvm, LLVM::TruncOp>,
        //     // Intrinsic_rint_llvm
        //     ToUnaryLLVMOpPattern<Intrinsic_sqrt_llvm, LLVM::SqrtOp>,
        //     // Intrinsic_sqrt_llvm_fast
        //     // Intrinsic_pointerref
        //     // Intrinsic_pointerset
        //     // Intrinsic_cglobal
        //     // Intrinsic_llvmcall
        //     // Intrinsic_arraylen
        //     // Intrinsic_cglobal_auto
        //     // Builtin_throw
        //     IsOpLowering, // Builtin_is
        //     // Builtin_typeof
        //     // Builtin_sizeof
        //     // Builtin_issubtype
        //     ToUndefOpPattern<Builtin_isa>, // Builtin_isa
        //     // Builtin__apply
        //     // Builtin__apply_pure
        //     // Builtin__apply_latest
        //     // Builtin__apply_iterate
        //     // Builtin_isdefined
        //     // Builtin_nfields
        //     // Builtin_tuple
        //     // Builtin_svec
        //     ToUndefOpPattern<Builtin_getfield>, // Builtin_getfield
        //     // Builtin_setfield
        //     // Builtin_fieldtype
        //     // Builtin_arrayref
        //     // Builtin_const_arrayref
        //     // Builtin_arrayset
        //     // Builtin_arraysize
        //     // Builtin_apply_type
        //     // Builtin_applicable
        //     // Builtin_invoke ?
        //     // Builtin__expr
        //     // Builtin_typeassert
        //     IfElseOpLowering // Builtin_ifelse
        //     // Builtin__typevar
        //     // invoke_kwsorter?
            >(&getContext(), converter);
        // patterns.insert<
        //     GotoOpLowering
        //     >(&getContext());

        if (failed(applyFullConversion(
                       getFunction(), target, patterns, &converter)))
            signalPassFailure();
    }
};

std::unique_ptr<Pass> mlir::jlir::createJLIRToLLVMLoweringPass() {
    return std::make_unique<JLIRToLLVMLoweringPass>();
}
