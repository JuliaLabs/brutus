#####
##### Builder
#####

# High-level version of MLIR's OpBuilder.

mutable struct JLIRBuilder
    ctx::JLIR.Context
    values::Vector{JLIR.Value}
    arguments::Vector{JLIR.Value}
    insertion::Int
    blocks::Vector{JLIR.Block}
    function JLIRBuilder()
        ctx = JLIR.create_context()
        ccall((:brutus_register_dialects, "libbrutus"),
              Cvoid,
              (JLIR.Context, ),
              ctx)
        new(ctx, JLIR.Value[], JLIR.Value[], 1)
    end
end

set_insertion!(b::JLIRBuilder, blk::Int) = b.insertion = blk

function push!(b::JLIRBuilder, op::JLIR.Operation)
    @assert(isdefined(b, :blocks))
    blk = b.blocks[b.insertion]
    push_operation!(blk, op)
end

#####
##### Utilities
#####

function convert_type_to_jlirtype(builder::JLIRBuilder, a)
    ctx = builder.ctx
    return ccall((:brutus_get_juliatype, "libbrutus"), 
                 JLIR.Type, 
                 (JLIR.Context, Any), 
                 ctx, a)
end

function convert_value_to_jlirattr(builder::JLIRBuilder, a)
    ctx = builder.ctx
    return ccall((:brutus_get_juliavalueattr, "libbrutus"), 
                 JLIR.Value, 
                 (JLIR.Context, Any), 
                 ctx, a)
end

function get_functype(builder::JLIRBuilder, args::Vector{JLIR.Type}, ret::JLIR.Type)
    return MLIR.API.mlirFunctionTypeGet(builder.ctx, length(args), args, 1, [ret])
end

function get_functype(builder::JLIRBuilder, args, ret)
    return get_functype(builder, length(args), map(args) do a
                            convert_type_to_jlirtype(builder, a)
                        end, 1, [convert_type_to_jlirtype(builder, ret)])
end

function unwrap(mi::Core.MethodInstance)
    return mi.def.value
end
unwrap(s) = s

function extract_linetable_meta(builder::JLIRBuilder, v::Vector{Core.LineInfoNode})
    locations = JLIR.Location[]
    for n in v
        method = unwrap(n.method)
        file = String(n.file)
        line = n.line
        inlined_at = n.inlined_at
        if method isa Method
            fname = String(method.name)
        end
        if method isa Symbol
            fname = String(method)
        end
        current = JLIR.Location(builder.ctx, fname, UInt32(line), UInt32(0)) # TODO: col.
        if inlined_at > 0
            current = JLIR.Location(current, locations[inlined_at - 1])
        end
        push!(locations, current)
    end
    return locations
end
