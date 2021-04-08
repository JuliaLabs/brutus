@enum DumpOption::UInt8 begin
    DumpIRCode        = 0
    DumpTranslated    = 1
    DumpCanonicalized = 2
    DumpLoweredToStd  = 4
    DumpLoweredToLLVM = 8
    DumpTranslateToLLVM = 16
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

#####
##### JIT
#####

const compiled_cache = Dict{UInt, Any}()

struct Thunk{F, RT, TT}
    f::F
    ptr::Ptr{Cvoid}
end

struct MLIRCompilerTarget <: AbstractCompilerTarget end
struct MLIRCompilerParams <: AbstractCompilerParams
    emit_fptr::Bool
    dump_options::Vector{DumpOption}
end

# Emit MLIR IR to stdout
function _emit(job::CompilerJob)
    ft, tt = job.source.f, job.source.tt.parameters
    emit_fptr = job.params.emit_fptr
    dump_options = job.params.dump_options

    # get first method instance matching signature
    entry_mi = get_methodinstance(Tuple{typeof(ft), tt...})
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

function _link(job::CompilerJob, (fptr, rt))
    f, tt = job.source.f, job.source.tt
    return Thunk{typeof(f), rt, tt}(f, fptr)
end

function emit(@nospecialize(f), tt...; emit_fptr = true, dump_options::Vector{DumpOption} = DumpOption[])
    fspec = FunctionSpec(f, Tuple{tt...}, false, nothing)
    job = CompilerJob(MLIRCompilerTarget(), fspec, MLIRCompilerParams(emit_fptr, dump_options))
    return GPUCompiler.cached_compilation(compiled_cache, job, _emit, _link)
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
    TT = map(Core.Typeof, args)
    return emit(f, TT...)(args...)
end