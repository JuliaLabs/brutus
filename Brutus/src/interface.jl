#####
##### GPUCompiler codegen
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

#####
##### Call Interface
#####

struct MemrefDescriptor{T, N}
    allocated :: Ptr{T}
    aligned   :: Ptr{T}
    offset    :: Int64
    size      :: NTuple{N, Int64}
    strides   :: NTuple{N, Int64}
end

function Base.convert(::Type{MemrefDescriptor{T, N}}, arr::Array{T, N}) where {T, N}
    allocated = pointer(arr)
    aligned   = pointer(arr)
    offset    = Int64(0)
    size      = Base.size(arr)
    strides   = (1, cumprod(size)[1:end-1]...)

    # colum-major to row-major
    size = reverse(size)
    strides = reverse(strides)

    MemrefDescriptor{T, N}(allocated, aligned, offset, size, strides)
end

struct Thunk{F, RT, TT}
    f::F
    ptr::Ptr{Cvoid}
end

const brutus_cache = Dict{UInt,Any}()

function link(job::CompilerJob, (fptr, rt))
    @assert fptr != C_NULL
    fptr, rt = result
    f = job.source.f
    tt = job.source.tt
    return Thunk{typeof(f), rt, tt}(f, fptr)
end

function thunk(f::F, tt::TT=Tuple{}; emit_fptr::Bool = true, dump_options::Vector{DumpOption} = DumpOption[]) where {F<:Base.Callable, TT<:Type}
    fspec = GPUCompiler.FunctionSpec(F, tt, false, nothing)
    target = BrutusCompilerTarget()
    params = BrutusCompilerParams(emit_fptr, dump_options)
    job = CompilerJob(target, fspec, params)
    return GPUCompiler.cached_compilation(brutus_cache, job, emit, link)
end

# Need to pass struct as pointer, to match cifacme ABI
abi(::Type{<:Array{T, N}}) where {T, N} = Ref{MemrefDescriptor{T, N}} 
function abi(T::DataType)
    if isprimitivetype(T)
        return T
    else
        return Any
    end
end

@inline (thunk::Thunk)(args...) = __call(thunk, args)

@generated function __call(thunk::Thunk{f, RT, TT}, args::TT) where {f, RT, TT}
    nargs = map(abi, (args.parameters...,))
    _args = (:(args[$i]) for i in 1:length(nargs))

    # Insert function type up-front
    nargs = (Any, nargs...)

    expr = quote
        ccall(thunk.ptr, $(abi(RT)), ($(nargs...),), thunk.f, $(_args...))::RT
    end
    return expr
end

function call(f::F, args...) where F
    TT = Tuple{map(Core.Typeof, args)...}
    return thunk(f, TT)(args...)
end
