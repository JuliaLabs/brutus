module Brutus

# Emit MLIR IR to stdout
function emit(f, tt...; optimize=0)
    name = (typeof(f) <: Function) ? nameof(f) : nameof(typeof(f))
    IR, rt = code_ircode(f, Tuple{tt...})[1]
    ccall((:brutus_codegen, "libbrutus"), Cvoid, (Any,Any,Cstring,Cint), IR, rt, name, optimize)
end

function code_ircode(@nospecialize(f), @nospecialize(types=Tuple);
                     world = Base.get_world_counter(),
                     params = Core.Compiler.Params(world))
    return [code_ircode(mi; world=world, params=params)
            for mi in Base.method_instances(f, types, world)]
end

function code_ircode(mi::Core.Compiler.MethodInstance;
                     world = Base.get_world_counter(),
                     params = Core.Compiler.Params(world))
    ccall(:jl_typeinf_begin, Cvoid, ())
    result = Core.Compiler.InferenceResult(mi)
    frame = Core.Compiler.InferenceState(result, false, params)
    frame === nothing && return nothing
    if Core.Compiler.typeinf(frame)
        opt = Core.Compiler.OptimizationState(frame)
        ir = Core.Compiler.run_passes(opt.src, opt.nargs - 1, opt)
        opt.src.inferred = true
    end
    ccall(:jl_typeinf_end, Cvoid, ())
    frame.inferred || return nothing
    return ir => Core.Compiler.widenconst(result.result)
end

end # module
