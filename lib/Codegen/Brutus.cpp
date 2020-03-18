#include "brutus/brutus.h"
#include "brutus/brutus_internal.h"
#include "brutus/Dialect/Julia/JuliaOps.h"

#include "mlir/InitAllDialects.h"

extern "C" {

jl_sym_t *call_sym;
jl_sym_t *invoke_sym;
jl_value_t *argument_type;
jl_value_t *const_type;
jl_value_t *gotoifnot_type;
jl_value_t *return_node_type;

void brutus_init() {
    // lookup session static data
    invoke_sym = jl_symbol("invoke");
    call_sym = jl_symbol("call");
    jl_module_t *core_module = (jl_module_t*)jl_get_global(
        jl_main_module, jl_symbol("Core"));
    jl_module_t *compiler_module = (jl_module_t*)jl_get_global(
        core_module, jl_symbol("Compiler"));
    argument_type = jl_get_global(compiler_module, jl_symbol("Argument"));
    const_type = jl_get_global(compiler_module, jl_symbol("Const"));
    gotoifnot_type = jl_get_global(compiler_module, jl_symbol("GotoIfNot"));
    return_node_type = jl_get_global(compiler_module, jl_symbol("ReturnNode"));

    mlir::registerAllDialects();
    // Register dialect
    mlir::registerDialect<mlir::jlir::JLIRDialect>();
}

} // extern "C"
