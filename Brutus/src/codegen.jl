#####
##### Codegen
#####

struct BrutusCompilerTarget <: AbstractCompilerTarget end
GPUCompiler.llvm_triple(::BrutusCompilerTarget) = Sys.MACHINE
GPUCompiler.llvm_machine(::BrutusCompilerTarget) = tm[]

module Runtime
    # the runtime library
    signal_exception() = return
    malloc(sz) = Base.Libc.malloc(sz)
    report_oom(sz) = return
    report_exception(ex) = return
    report_exception_name(ex) = return
    report_exception_frame(idx, func, file, line) = return
end

@enum DumpOption::UInt8 begin
    DumpIRCode        = 0
    DumpTranslated    = 1
    DumpCanonicalized = 2
    DumpLoweredToStd  = 4
    DumpLoweredToLLVM = 8
    DumpTranslateToLLVM = 16
end

struct BrutusCompilerParams <: AbstractCompilerParams
    emit_fptr::Bool
    dump_options::Vector{DumpOption}
end

GPUCompiler.ci_cache(job::CompilerJob{BrutusCompilerTarget}) = GLOBAL_CI_CACHE
GPUCompiler.runtime_module(job::CompilerJob{BrutusCompilerTarget}) = Runtime
GPUCompiler.isintrinsic(::CompilerJob{BrutusCompilerTarget}, fn::String) = true
GPUCompiler.can_throw(::CompilerJob{BrutusCompilerTarget}) = true
GPUCompiler.runtime_slug(job::CompilerJob{BrutusCompilerTarget}) = "brutus"

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
function emit(job::CompilerJob)
    ft = job.source.f
    tt = job.source.tt
    emit_fptr = job.params.emit_fptr
    dump_options = job.params.dump_options
    name = (ft <: Function) ? nameof(ft.instance) : nameof(ft)

    # get first method instance matching signature
    entry_mi = get_methodinstance(Tuple{ft, tt.parameters...})
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
    return (fptr, rt)
end

function emit(@nospecialize(ft), @nospecialize(tt);
              emit_fptr::Bool=true,
              dump_options::Vector{DumpOption}=DumpOption[])
    fspec = GPUCompiler.FunctionSpec(ft, Tuple{tt...}, false, nothing)
    target = BrutusCompilerTarget()
    params = BrutusCompilerParams(emit_fptr, dump_options)
    job = CompilerJob(target, fspec, params)
    return emit(job)
end
