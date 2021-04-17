const GLOBAL_CI_CACHE = GPUCompiler.CodeCache()

function __init__()
    ccall((:brutus_init, "libbrutus"), Cvoid, (Any,), @__MODULE__)
end

