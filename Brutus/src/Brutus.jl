module Brutus

function __init__()
    ccall((:brutus_init, "libbrutus"), Cvoid, ())
end

@enum DumpOption::UInt8 begin
    DumpIRCode        = 0
    DumpTranslated    = 1
    DumpCanonicalized = 2
    DumpLoweredToStd  = 4
    DumpLoweredToLLVM = 8
end

function find_invokes(IR)
    callees = Core.MethodInstance[]
    for stmt in IR.stmts
        if stmt isa Expr
            if stmt.head == :invoke
                mi = stmt.args[1]
                push!(callees, mi)
            end
        end
    end
    return callees
end

# Emit MLIR IR to stdout
function emit(@nospecialize(ft), @nospecialize(tt);
              emit_fptr::Bool=true,
              dump_options::Vector{DumpOption}=DumpOption[])
    name = (ft <: Function) ? nameof(ft.instance) : nameof(ft)

    # get first method instance matching signature
    entry_mi = get_methodinstance(Tuple{ft, tt...})
    IR, rt = code_ircode(entry_mi)

    if DumpIRCode in dump_options
        println("return type: ", rt)
        println("IRCode:\n")
        println(IR)
    end

    worklist = [IR]
    methods = Dict{Core.MethodInstance, Tuple{Core.Compiler.IRCode, Any}}(
        entry_mi => (IR, rt)
    )

    while !isempty(worklist)
        code = pop!(worklist)
        callees = find_invokes(code)
        for callee in callees
            if !haskey(methods, callee)
                _code, _rt = code_ircode(callee)

                methods[callee] = (_code, _rt)
                push!(worklist, _code)
            end
        end
    end

    # generate LLVM bitcode and load it
    dump_flags = reduce(|, map(UInt8, dump_options), init=0)
    fptr = ccall((:brutus_codegen, "libbrutus"),
                 Ptr{Nothing},
                 (Any, Any, Cuchar, Cuchar),
                 methods, entry_mi, emit_fptr, dump_flags)
    if convert(Int, fptr) == 0
        return nothing
    end

    return fptr, rt
end

@generated function call(f, args...)
    result = emit(f, args)
    if result === nothing
        error("failed to emit function")
    end
    fptr, rt, = result
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

function get_methodinstance(@nospecialize(sig);
                            world=Base.get_world_counter(),
                            interp=Core.Compiler.NativeInterpreter(world))
    ms = Base._methods_by_ftype(sig, 1, Base.get_world_counter())
    @assert length(ms) == 1
    m = ms[1]
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
    frame = Core.Compiler.InferenceState(result, false, interp)
    frame === nothing && return nothing
    if Core.Compiler.typeinf(interp, frame)
        opt_params = Core.Compiler.OptimizationParams(interp)
        opt = Core.Compiler.OptimizationState(frame, opt_params, interp)
        ir = Core.Compiler.run_passes(opt.src, opt.nargs - 1, opt)
        opt.src.inferred = true
    end
    ccall(:jl_typeinf_end, Cvoid, ())
    frame.inferred || return nothing
    # TODO(yhls): Fix this upstream
    resize!(ir.argtypes, opt.nargs)
    return ir => Core.Compiler.widenconst(result.result)
end

include("reflection.jl")

end # module
