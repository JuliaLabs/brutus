#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "brutus/Dialect/Julia/JuliaOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
//using namespace mlir::linalg;
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
struct LowerIntrinsicCallPattern : public OpRewritePattern<jlir::CallOp> {
    public:
        using OpRewritePattern<jlir::CallOp>::OpRewritePattern;

    LogicalResult match(jlir::CallOp op) const override {
        Value callee = op.callee();
        Operation *definingOp = callee.getDefiningOp();
        if (!definingOp) {
            // Value is block-argument.
            return failure();
        }
        if (jlir::ConstantOp constant = dyn_cast<jlir::ConstantOp>(definingOp)) {
            jl_value_t* value = constant.value();
            if (jl_typeis(value, jl_intrinsic_type)) {
                return success();
            }
        }
        return failure();
    }

    void rewrite(jlir::CallOp op, PatternRewriter &rewriter) const override {
        jlir::ConstantOp defining = dyn_cast<jlir::ConstantOp>(op.callee().getDefiningOp());
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
struct LowerBuiltinCallPattern : public OpRewritePattern<jlir::CallOp> {
    public:
        using OpRewritePattern<mlir::jlir::CallOp>::OpRewritePattern;

    LogicalResult match(jlir::CallOp op) const override {
        Value callee = op.callee();
        Operation *definingOp = callee.getDefiningOp();
        if (!definingOp) {
            // Value is block-argument.
            return failure();
        }
        if (auto constant = dyn_cast<jlir::ConstantOp>(definingOp)) {
            jl_value_t* value = constant.value();
            if (jl_isa(value, (jl_value_t*)jl_builtin_type)) {
                return success();
            }
        }
        return failure();
    }

    void rewrite(jlir::CallOp op, PatternRewriter &rewriter) const override {
        auto defining = dyn_cast<jlir::ConstantOp>(op.callee().getDefiningOp());
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

/// Intrinsic rewriter
struct LowerCustomIntrinsicCallPattern : public OpRewritePattern<InvokeOp> {
    public:
        using OpRewritePattern<InvokeOp>::OpRewritePattern;

    LogicalResult match(InvokeOp op) const override {
        Value callee = op.callee();
        Operation *definingOp = callee.getDefiningOp();
        llvm::errs() << " trying to lower: " << op << "\n";
        if (!definingOp) {
            llvm::errs() << " not an op\n";
            // Value is block-argument.
            return failure();
        }
        if (jlir::ConstantOp constant = dyn_cast<jlir::ConstantOp>(definingOp)) {
            Type resultType = op.getResult().getType();
            if (auto jlResultType = resultType.dyn_cast<JuliaType>()) {
                jl_((jl_value_t*)jlResultType.getDatatype());
                if (jl_subtype((jl_value_t*)jlResultType.getDatatype(), br_intrinsic_type)) {
                    return success();
                }
                llvm::errs() << " not typeis\n";
            }
            llvm::errs() << " not JLType\n";
        }
        llvm::errs() << " not constant : " << *definingOp << "\n";

        return failure();
    }

    void rewrite(InvokeOp op, PatternRewriter &rewriter) const override {
        jlir::ConstantOp defining = dyn_cast<jlir::ConstantOp>(op.callee().getDefiningOp());

        auto args = op.arguments();
        // Replace the value of the old Op with the Result of the new op
        SmallVector<mlir::Value, 2> inargs = {args[2], args[3]};
        SmallVector<mlir::Value, 1> outargs = {args[1]};
        rewriter.replaceOpWithNewOp<mlir::linalg::MatmulOp>(op, inargs, outargs);
    }
};

} // namespace

/// Register our patterns as "canonicalization" patterns on the jlir::CallOp so
/// that they can be picked up by the Canonicalization framework.
void jlir::CallOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                         MLIRContext *context) {
    results.insert<LowerIntrinsicCallPattern>(context);
    results.insert<LowerBuiltinCallPattern>(context);
    results.insert<LowerCustomIntrinsicCallPattern>(context);
}

namespace {

struct SimplifyRedundantConvertStdOps : public OpRewritePattern<ConvertStdOp> {
    using OpRewritePattern<ConvertStdOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(ConvertStdOp op,
                                  PatternRewriter &rewriter) const override {
        // 1. Check if input = output type
        Type resultType = op.getResult().getType();
        Type inputType = op.getOperand().getType();
        if (resultType == inputType) {
            rewriter.replaceOp(op, {op.getOperand()});
            return success();
        }
        
        // 2. Check if we have a chain of convert ops
        ConvertStdOp inputOp = dyn_cast_or_null<ConvertStdOp>(
            op.getOperand().getDefiningOp());
        if (!inputOp)
            return failure();

        // if the chain roundtrips elimnate it.
        Type originalType = inputOp.getOperand().getType();
        if (originalType == resultType) {
            rewriter.replaceOp(op, {inputOp.getOperand()});
            return success();
        }

        // 3. Shortcut the input op.
        rewriter.replaceOpWithNewOp<ConvertStdOp>(op, resultType, inputOp.getOperand());
        return success(); 
    }
};

} // namespace

void ConvertStdOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                               MLIRContext *context) {
    results.insert<SimplifyRedundantConvertStdOps>(context);
}
