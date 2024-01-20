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
    Optional<Type> convertJuliaType(JuliaType t);
    Type convertBitstype(jl_datatype_t *jdt);
};

/// Collect the patterns to convert from the JLIR dialect to Standard The
/// conversion patterns capture the LLVMTypeConverter and the LowerToLLVMOptions
/// by reference meaning the references have to remain alive during the entire
/// pattern lifetime.
void populateJLIRToStdConversionPatterns(RewritePatternSet &patterns,
                                         MLIRContext &context,
                                         JLIRToStandardTypeConverter &converter);

/// Create a pass to convert JLIR operations to the Standard dialect.
std::unique_ptr<OperationPass<ModuleOp>> createJLIRToStandardLoweringPass();

} // namespace jlir
} // namespace mlir

#endif // JLIR_CONVERSION_JLIRTOSTANDARD_JLIRTOSTANDARD_H_
