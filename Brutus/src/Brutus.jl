module Brutus

using Preferences
using MLIR

const libbrutus = @load_preference("libbrutus")
const libbrutus_c = @load_preference("libbrutus_c")

module BrutusAPI
    import ..Brutus: libbrutus, libbrutus_c
    import MLIR.API: MlirDialectHandle
    function mlirGetDialectHandle__jlir__()
        @ccall libbrutus_c.mlirGetDialectHandle__jlir__()::MlirDialectHandle
    end
end

import MLIR: API, IR, Dialects
module BrutusDialects
    include(joinpath("Dialects", string(Base.libllvm_version.major), "JuliaOps.jl"))
end

function load_dialect(ctx)
    dialect = IR.DialectHandle(BrutusAPI.mlirGetDialectHandle__jlir__())
    API.mlirDialectHandleRegisterDialect(dialect, ctx)
    API.mlirDialectHandleLoadDialect(dialect, ctx)
end

function code_mlir(f, types)
    ctx = IR.context()

    src, rt = only(Base.code_ircode(f, types))

    for dialect in ("func", "cf")
        IR.get_or_load_dialect!(dialect)
    end
    load_dialect(ctx)
    IR.get_or_load_dialect!(IR.DialectHandle(BrutusAPI.mlirGetDialectHandle__jlir__()))

    values = Vector{Value}(undef, length(ir.stmts))
end

# using GPUCompiler: GPUCompiler, CompilerJob
# import GPUCompiler
# import GPUCompiler: AbstractCompilerTarget, AbstractCompilerParams

# #####
# ##### Exports
# #####

# export emit

# include("init.jl")
# include("codegen.jl")
# include("reflection.jl")
# include("interface.jl")

end # module
