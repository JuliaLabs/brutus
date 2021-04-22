#####
##### Builder
#####

# High-level version of MLIR's OpBuilder.

struct JLIRBuilder
    ctx::JLIR.Context
    insertion::Ref{Int}
    values::Vector{JLIR.Value}
    arguments::Vector{JLIR.Type}
    locations::Vector{JLIR.Location}
    blocks::Vector{JLIR.Block}
    code::Core.Compiler.IRCode
    state::JLIR.OperationState
end

function JLIRBuilder(code::Core.Compiler.IRCode, name::String)
    ctx = JLIR.create_context()
    ccall((:brutus_register_dialects, "libbrutus"),
          Cvoid,
          (JLIR.Context, ),
          ctx)
    irstream = code.stmts
    stmts = irstream.inst
    types = irstream.type
    location_indices = getfield(irstream, :line)
    linetable = getfield(code, :linetable)
    locations = extract_linetable_locations(ctx, linetable)
    argtypes = getfield(code, :argtypes)
    args = [convert_type_to_jlirtype(ctx, a) for a in argtypes]
    state = JLIR.create_operation_state(name, locations[1])
    entry_blk, reg = JLIR.add_entry_block!(state, args)
    tr = JLIR.get_first_block(reg)
    nblocks = length(code.cfg.blocks)
    blocks = JLIR.Block[entry_blk]
    for i in 1 : nblocks
        blk = JLIR.Block()
        JLIR.push!(reg, blk)
        push!(blocks, blk)
    end
    return JLIRBuilder(ctx, Ref(2), JLIR.Value[], args, locations, blocks, code, state)
end

set_insertion!(b::JLIRBuilder, blk::Int) = b.insertion[] = blk
get_insertion_block(b::JLIRBuilder) = b.blocks[b.insertion[]]

get_stmts(b::JLIRBuilder) = b.code.stmts.inst
get_types(b::JLIRBuilder) = b.code.stmts.type
get_stmt(b::JLIRBuilder, ind::Int) = getindex(b.code.stmts.inst, ind)
get_type(b::JLIRBuilder, ind::Int) = getindex(b.code.stmts.type, ind)
get_cfg(b::JLIRBuilder) = b.code.cfg

function push!(b::JLIRBuilder, op::JLIR.Operation)
    blk = b.blocks[b.insertion]
    push_operation!(blk, op)
end

finish(b::JLIRBuilder) = JLIR.Operation(b.state)

#####
##### Utilities
#####

function convert_type_to_jlirtype(ctx::JLIR.Context, a)
    return ccall((:brutus_get_jlirtype, "libbrutus"), 
                 JLIR.Type, 
                 (JLIR.Context, Any), 
                 ctx, a)
end

function convert_value_to_jlirattr(ctx::JLIR.Context, a)
    return ccall((:brutus_get_jlirattr, "libbrutus"), 
                 JLIR.Attribute, 
                 (JLIR.Context, Any), 
                 ctx, a)
end

function get_functype(builder::JLIRBuilder, args::Vector{JLIR.Type}, ret::JLIR.Type)
    return MLIR.API.mlirFunctionTypeGet(builder.ctx, length(args), args, 1, [ret])
end

function get_functype(builder::JLIRBuilder, args, ret)
    return get_functype(builder, length(args), map(args) do a
                            convert_type_to_jlirtype(builder.ctx, a)
                        end, 1, [convert_type_to_jlirtype(builder.ctx, ret)])
end

function unwrap(mi::Core.MethodInstance)
    return mi.def.value
end
unwrap(s) = s

function extract_linetable_locations(ctx::JLIR.Context, v::Vector{Core.LineInfoNode})
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
        current = JLIR.Location(ctx, fname, UInt32(line), UInt32(0)) # TODO: col.
        if inlined_at > 0
            current = JLIR.Location(current, locations[inlined_at - 1])
        end
        push!(locations, current)
    end
    return locations
end
