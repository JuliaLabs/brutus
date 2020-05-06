module Brutus

import LLVM

function __init__()
    ccall((:brutus_init, "libbrutus"), Cvoid, ())
end

@enum DumpOption::UInt8 begin
    DumpIRCode        = 0
    DumpTranslated    = 1
    DumpOptimized     = 2
    DumpLoweredToStd  = 4
    DumpLoweredToLLVM = 8
end

# Emit MLIR IR to stdout
function emit(@nospecialize(ft), @nospecialize(tt);
              emit_fptr::Bool=true,
              optimize::Bool=true,
              dump_options::Vector{DumpOption}=DumpOption[])
    name = (ft <: Function) ? nameof(ft.instance) : nameof(ft)

    # get first IRCode matching signature
    matches = code_ircode_by_signature(Tuple{ft, tt...})
    @assert length(matches) > 0 "no method instances matching given signature"
    IR, rt = first(matches)

    if DumpIRCode in dump_options
        println("return type: ", rt)
        println("IRCode:\n")
        println(IR)
    end

    # generate LLVM bitcode and load it
    dump_flags = reduce(|, map(UInt8, dump_options), init=0)
    fptr = ccall((:brutus_codegen, "libbrutus"),
                 Ptr{Nothing},
                 (Any, Any, Cstring, Cuchar, Cuchar, Cuchar),
                 IR, rt, name, emit_fptr, optimize, dump_flags)
    if convert(Int, fptr) == 0
        return nothing
    end

    return fptr, rt
end

@generated function call(f, args...)
    fptr, rt, = emit(f, args)
    convert_type(t) = isprimitivetype(t) ? t : Any
    converted_args = collect(map(convert_type, args))
    return quote
        arg_pointers = Ref[
            Base.unsafe_convert(Ptr{Cvoid}, Ref{$(convert_type(f))}(f))]
        for i in 1:length(args)
            push!(arg_pointers,
                  Base.unsafe_convert(
                      Ptr{Cvoid}, Ref{$converted_args[i]}(args[i])))
        end
        push!(arg_pointers, Ref{$(convert_type(rt))}())
        # ccall(:jl_breakpoint, Cvoid, (Ptr{Ptr{Cvoid}},), arg_pointers)
        ccall($fptr, Cvoid, (Ptr{Ptr{Cvoid}},), arg_pointers)
        arg_pointers[end][]
    end
end

function code_ircode_by_signature(@nospecialize(sig);
                                  world=Base.get_world_counter(),
                                  params=Core.Compiler.Params(world))
    return [code_ircode(ccall(:jl_specializations_get_linfo,
                              Ref{Core.MethodInstance},
                              (Any, Any, Any),
                              data[3], data[1], data[2]);
                        world=world, params=params)
            for data in Base._methods_by_ftype(sig, -1, world)]
end

function code_ircode(@nospecialize(f), @nospecialize(types=Tuple);
                     world=Base.get_world_counter(),
                     params=Core.Compiler.Params(world))
    return [code_ircode(mi; world=world, params=params)
            for mi in Base.method_instances(f, types, world)]
end

function code_ircode(mi::Core.Compiler.MethodInstance;
                     world=Base.get_world_counter(),
                     params=Core.Compiler.Params(world))
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
    # TODO(yhls): Fix this upstream
    resize!(ir.argtypes, opt.nargs)
    return ir => Core.Compiler.widenconst(result.result)
end

end # module
