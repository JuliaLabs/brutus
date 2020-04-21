#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "brutus/Dialect/Julia/JuliaOps.h"

#include "julia.h"
#include "juliapriv/julia_private.h"

namespace JL_I {
#include "juliapriv/intrinsics.h"
}

#include "juliapriv/builtin_proto.h"

using namespace mlir;
using namespace jlir;

namespace {

/// Intrinsic rewriter
struct LowerIntrinsicCallPattern : public OpRewritePattern<CallOp> {
    public:
        using OpRewritePattern<CallOp>::OpRewritePattern;

    LogicalResult match(CallOp op) const override {
        Value callee = op.callee();
        Operation *definingOp = callee.getDefiningOp();
        if (!definingOp) {
            // Value is block-argument.
            return failure();
        }
        if (ConstantOp constant = dyn_cast<ConstantOp>(definingOp)) {
            jl_value_t* value = constant.value();
            if (jl_typeis(value, jl_intrinsic_type)) {
                return success();
            }
        }
        return failure();
    }

    void rewrite(CallOp op, PatternRewriter &rewriter) const override {
        ConstantOp defining = dyn_cast<ConstantOp>(op.callee().getDefiningOp());
        JL_I::intrinsic f = (JL_I::intrinsic)*(uint32_t*)jl_data_ptr(defining.value());
        assert(f < JL_I::num_intrinsics);
        StringRef name = "jlir." + std::string(jl_intrinsic_name(f));

        // The IntrinsicOps are defined by their name, so we can lookup the name
        // from Julia and construct a generic Operation based on an OperationState
        OperationState state = OperationState(op.getLoc(), name);
        state.addOperands(op.arguments());
        state.addTypes(op.getType());
        Operation *newOp = rewriter.createOperation(state);
        // Replace the value of the old Op with the Result of the new op
        rewriter.replaceOp(op, newOp->getResult(0));
    }
};

/// Builtin rewriter
struct LowerBuiltinCallPattern : public OpRewritePattern<CallOp> {
    public:
        using OpRewritePattern<CallOp>::OpRewritePattern;

    LogicalResult match(CallOp op) const override {
        Value callee = op.callee();
        Operation *definingOp = callee.getDefiningOp();
        if (!definingOp) {
            // Value is block-argument.
            return failure();
        }
        if (ConstantOp constant = dyn_cast<ConstantOp>(definingOp)) {
            jl_value_t* value = constant.value();
            if (jl_isa(value, (jl_value_t*)jl_builtin_type)) {
                return success();
            }
        }
        return failure();
    }

    void rewrite(CallOp op, PatternRewriter &rewriter) const override {
        ConstantOp defining = dyn_cast<ConstantOp>(op.callee().getDefiningOp());
        assert(jl_isa(defining.value(), (jl_value_t*)jl_builtin_type));
        jl_datatype_t* typeof_builtin = (jl_datatype_t*)jl_typeof(defining.value());
        StringRef name = "jlir." + std::string(jl_symbol_name(typeof_builtin->name->mt->name));

        // TODO: factor out
        OperationState state = OperationState(op.getLoc(), name);
        state.addOperands(op.arguments());
        state.addTypes(op.getType());
        Operation *newOp = rewriter.createOperation(state);
        // Replace the value of the old Op with the Result of the new op
        rewriter.replaceOp(op, newOp->getResult(0));
    }
};

} // namespace

/// Register our patterns as "canonicalization" patterns on the CallOp so
/// that they can be picked up by the Canonicalization framework.
void CallOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                         MLIRContext *context) {
    results.insert<LowerIntrinsicCallPattern>(context);
    results.insert<LowerBuiltinCallPattern>(context);
}

namespace {

struct SimplifyRedundantConvertStdOps : public OpRewritePattern<ConvertStdOp> {
    using OpRewritePattern<ConvertStdOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(ConvertStdOp op,
                                  PatternRewriter &rewriter) const override {
        ConvertStdOp inputOp = dyn_cast_or_null<ConvertStdOp>(
            op.getOperand().getDefiningOp());

        if (!inputOp)
            return failure();

        assert(inputOp.getOperand().getType() == op.getResult().getType());

        rewriter.replaceOp(op, {inputOp.getOperand()});
        return success();
    }
};

} // namespace

void ConvertStdOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                               MLIRContext *context) {
    results.insert<SimplifyRedundantConvertStdOps>(context);
}
