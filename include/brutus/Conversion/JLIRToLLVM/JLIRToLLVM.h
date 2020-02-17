#ifndef JLIR_CONVERSION_JLIRTOLLVM_JLIRTOLLVM_H_
#define JLIR_CONVERSION_JLIRTOLLVM_JLIRTOLLVM_H_

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class Pass;
namespace jlir {

/// Create a pass to convert Linalg operations to the LLVMIR dialect.
std::unique_ptr<Pass> createJLIRToLLVMLoweringPass();

} // namespace jlir
} // namespace mlir

#endif // JLIR_CONVERSION_JLIRTOLLVM_JLIRTOLLVM_H_
