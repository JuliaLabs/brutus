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
