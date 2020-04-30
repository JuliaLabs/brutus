#ifndef JLIR_CONVERSION_JLIRTOSTANDARD_JLIRTOSTANDARD_H_
#define JLIR_CONVERSION_JLIRTOSTANDARD_JLIRTOSTANDARD_H_

#include "brutus/Dialect/Julia/JuliaOps.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace jlir {

struct JLIRToStandardTypeConverter : public TypeConverter {
    MLIRContext *ctx;

    JLIRToStandardTypeConverter(MLIRContext *ctx);
    Optional<Type> convert_JuliaType(JuliaType t);
    Type convert_bitstype(jl_datatype_t *jdt);

    Operation *materializeConversion(PatternRewriter &rewriter,
                                     Type resultType,
                                     ArrayRef<Value> inputs,
                                     Location loc) override;
};

struct JLIRToStandardLoweringPass
    : public PassWrapper<JLIRToStandardLoweringPass, FunctionPass> {

    static bool isFuncOpLegal(FuncOp op, JLIRToStandardTypeConverter &converter);
    void runOnFunction() final;
};

/// Create a pass to convert JLIR operations to the Standard dialect.
std::unique_ptr<Pass> createJLIRToStandardLoweringPass();

// HACK
struct JLIRToStandardHackTypeConverter : public TypeConverter {
    MLIRContext *ctx;

    JLIRToStandardHackTypeConverter(MLIRContext *ctx);
};

struct JLIRToStandardHackLoweringPass
    : public PassWrapper<JLIRToStandardHackLoweringPass, FunctionPass> {

    static bool isFuncOpLegal(FuncOp op);
    void runOnFunction() final;
};

std::unique_ptr<Pass> createJLIRToStandardHackLoweringPass();

} // namespace jlir
} // namespace mlir

#endif // JLIR_CONVERSION_JLIRTOSTANDARD_JLIRTOSTANDARD_H_
