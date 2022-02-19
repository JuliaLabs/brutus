function get_methodinstance(@nospecialize(sig);
                            world=Base.get_world_counter(),
                            interp=Core.Compiler.NativeInterpreter(world))
    ms = Base._methods_by_ftype(sig, 1, Base.get_world_counter())
    @assert length(ms) == 1
    m = ms[1]
    display(m)
    mi = ccall(:jl_specializations_get_linfo,
               Ref{Core.MethodInstance}, (Any, Any, Any),
               m[3], m[1], m[2])
    return mi
end

function code_ircode_by_signature(@nospecialize(sig);
                                  world=Base.get_world_counter(),
                                  interp=Core.Compiler.NativeInterpreter(world))
    return [code_ircode(ccall(:jl_specializations_get_linfo,
                              Ref{Core.MethodInstance},
                              (Any, Any, Any),
                              data[3], data[1], data[2]);
                        world=world, interp=interp)
            for data in Base._methods_by_ftype(sig, -1, world)]
end

function code_ircode(@nospecialize(f), @nospecialize(types=Tuple);
                     world=Base.get_world_counter(),
                     interp=Core.Compiler.NativeInterpreter(world))
    return [code_ircode(mi; world=world, interp=interp)
            for mi in Base.method_instances(f, types, world)]
end

function code_ircode(mi::Core.Compiler.MethodInstance;
                     world=Base.get_world_counter(),
                     interp=Core.Compiler.NativeInterpreter(world))
    ccall(:jl_typeinf_begin, Cvoid, ())
    result = Core.Compiler.InferenceResult(mi)
@static if VERSION >= v"1.8.0-DEV.472"
    cache = :local
else
    cache = false
end
    frame = Core.Compiler.InferenceState(result, cache, interp)
    frame === nothing && return nothing
    if Core.Compiler.typeinf(interp, frame)
        opt_params = Core.Compiler.OptimizationParams(interp)
        opt = Core.Compiler.OptimizationState(frame, opt_params, interp)
@static if VERSION >= v"1.8.0-DEV.1570"
        ir = Core.Compiler.run_passes(opt.src, opt, frame.result)
else
        ir = Core.Compiler.run_passes(opt.src, opt.nargs - 1, opt)
end
        opt.src.inferred = true
    end
    ccall(:jl_typeinf_end, Cvoid, ())
    frame.inferred || return nothing
#     resize!(ir.argtypes, opt.nargs)
    return ir => Core.Compiler.widenconst(result.result)
end

emit_optimized(f, tt...) =
    emit(typeof(f), tt,
         emit_fptr=false,
         dump_options=[Brutus.DumpCanonicalized])

emit_lowered(f, tt...) =
    emit(typeof(f), tt,
         emit_fptr=false, # TODO: change to true when ready
         dump_options=[Brutus.DumpLoweredToStd,
                       Brutus.DumpLoweredToLLVM])

emit_translated(f, tt...) =
    emit(typeof(f), tt,
         emit_fptr=false,
         dump_options=[Brutus.DumpTranslated])

emit_llvm(f, tt...) =
    emit(typeof(f), tt,
         emit_fptr=false, # TODO: change to true when ready
         dump_options=[Brutus.DumpLoweredToLLVM,
                       Brutus.DumpTranslateToLLVM])

function lit(emit_sym)
    @assert length(ARGS) == 1
    temp = Module(:Main)
    expr = quote
            include(fname) = Core.include(@__MODULE__, fname)
            import Brutus
            emit(args...;kwargs...) = Brutus.$(emit_sym)(args...;kwargs...)
            include($(ARGS[1]))
    end
    Base.eval(temp, Expr(:toplevel, expr))
end
