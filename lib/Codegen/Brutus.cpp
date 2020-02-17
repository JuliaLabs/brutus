#include "brutus/brutus.h"
#include "brutus/Dialect/Julia/JuliaOps.h"
#include "mlir/InitAllDialects.h"

extern "C" {
jl_sym_t *call_sym;    jl_sym_t *invoke_sym;

void brutus_init() {
    // lookup session static data
    invoke_sym = jl_symbol("invoke");
    call_sym = jl_symbol("call");

    mlir::registerAllDialects();
    // Register dialect
    mlir::registerDialect<mlir::jlir::JLIRDialect>();
}
} // extern "C"

