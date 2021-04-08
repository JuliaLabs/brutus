module Brutus

using GPUCompiler
import GPUCompiler: FunctionSpec, AbstractCompilerTarget, AbstractCompilerParams

function __init__()
    ccall((:brutus_init, "libbrutus"), Cvoid, (Any,), @__MODULE__)
end

include("intrinsics.jl")
include("reflection.jl")
include("interface.jl")

end # module
