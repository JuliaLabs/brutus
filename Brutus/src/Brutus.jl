module Brutus

using GPUCompiler: GPUCompiler, CompilerJob
import GPUCompiler
import GPUCompiler: AbstractCompilerTarget, AbstractCompilerParams

#####
##### Exports
#####

export emit

include("init.jl")
include("compiler/Compiler.jl")
include("reflection.jl")
include("interface.jl")

end # module
