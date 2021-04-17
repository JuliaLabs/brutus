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
    fspec = GPUCompiler.FunctionSpec(f, tt, false, nothing)
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
