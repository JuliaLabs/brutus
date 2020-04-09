#ifndef JLIR_CONVERSION_JLIRTOSTANDARD_JLIRTOSTANDARD_H_
#define JLIR_CONVERSION_JLIRTOSTANDARD_JLIRTOSTANDARD_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace jlir {

/// Create a pass to convert JLIR operations to the Standard dialect.
std::unique_ptr<Pass> createJLIRToStandardLoweringPass();

} // namespace jlir
} // namespace mlir

#endif // JLIR_CONVERSION_JLIRTOSTANDARD_JLIRTOSTANDARD_H_
