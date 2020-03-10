module Brutus

import LLVM

function __init__()
    ccall((:brutus_init, "libbrutus"), Cvoid, ())
end

@enum DumpOption::UInt8 begin
    DumpTranslated = 1
    DumpOptimized  = 2
    DumpLowered    = 4 # after lowering to LLVM dialect
    DumpLLVMIR     = 8
end

# Emit MLIR IR to stdout
function emit(@nospecialize(ft), @nospecialize(tt);
              emit_llvm::Bool=true,
              optimize::Bool=true,
              dump_options::Vector{DumpOption}=DumpOption[])
    name = (ft <: Function) ? nameof(ft.instance) : nameof(ft)

    # get first IRCode matching signature
    matches = code_ircode_by_signature(Tuple{ft, tt...})
    @assert length(matches) > 0 "no method instances matching given signature"
    IR, rt = first(matches)

    # generate LLVM bitcode and load it
    dump_flags = reduce(|, map(UInt8, dump_options), init=0)
    bitcode = ccall((:brutus_codegen, "libbrutus"),
                    LLVM.MemoryBuffer,
                    (Any, Any, Cstring, Cuchar, Cuchar, Cuchar),
                    IR, rt, name, emit_llvm, optimize, dump_flags)
    if !emit_llvm
        return nothing
    end
    @assert convert(UInt64, LLVM.ref(bitcode)) != 0 "Brutus codegen failed"
    module_ref = Ref{LLVM.API.LLVMModuleRef}()
    context = LLVM.Interop.JuliaContext()
    status = convert(Bool, LLVM.API.LLVMParseBitcodeInContext2(
        LLVM.ref(context), LLVM.ref(bitcode), module_ref))
    @assert !status # caught by LLVM.jl diagnostics handler
    mod = LLVM.Module(module_ref[])

    # add attributes to LLVM function so that it inlines
    llvm_function = LLVM.functions(mod)[string(name)]
    push!(LLVM.function_attributes(llvm_function),
          LLVM.EnumAttribute("alwaysinline", 0, context))
    LLVM.linkage!(llvm_function, LLVM.API.LLVMPrivateLinkage)

    return llvm_function, rt
end

@generated function call(f, args...)
    llvm_function, rt, = emit(f, args)
    _args = (:(args[$i]) for i in 1:length(args))
    LLVM.Interop.call_function(llvm_function,
                               rt, Tuple{args...},
                               Expr(:tuple, _args...))
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
