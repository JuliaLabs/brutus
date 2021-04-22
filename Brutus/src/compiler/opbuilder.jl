#####
##### Builder
#####

# High-level version of MLIR's OpBuilder.

mutable struct JLIRBuilder
    ctx::JLIR.Context
    insertion::Int
    state::JLIR.OperationState
    values::Vector{JLIR.Value}
    arguments::Vector{JLIR.Type}
    locations::Vector{JLIR.Location}
    blocks::Vector{JLIR.Block}
    code::Core.Compiler.IRCode
    function JLIRBuilder()
        ctx = JLIR.create_context()
        ccall((:brutus_register_dialects, "libbrutus"),
              Cvoid,
              (JLIR.Context, ),
              ctx)
        new(ctx, 1)
    end
end

function JLIRBuilder(code::Core.Compiler.IRCode, name::String)
    b = JLIRBuilder()
    irstream = code.stmts
    stmts = irstream.inst
    types = irstream.type
    location_indices = getfield(irstream, :line)
    linetable = getfield(code, :linetable)
    locations = extract_linetable_meta(b, linetable)
    argtypes = getfield(code, :argtypes)
    args = [convert_type_to_jlirtype(b, a) for a in argtypes]
    state = JLIR.create_operation_state(name, locations[1])
    entry_blk, reg = JLIR.add_entry_block!(state, args)
    tr = JLIR.get_first_block(reg)
    nblocks = length(code.cfg.blocks)
    blocks = JLIR.Block[entry_blk]
    for i in 1 : nblocks
        blk = JLIR.Block()
        JLIR.insertafter!(reg, entry_blk, blk)
        push!(blocks, blk)
    end
    b.state = state
    b.arguments = args
    b.locations = locations
    b.blocks = blocks
    b.state = state
    b.code = code
    b.arguments = args
    v = walk_cfg_emit_branchargs(b, 1, 2, locations[1])
    goto = create!(b, GotoOp(), JLIR.Location(b.ctx), blocks[2], v)
    set_insertion!(b, 2)
    return b
end

set_insertion!(b::JLIRBuilder, blk::Int) = setfield!(b, :insertion, blk)
get_insertion_block(b::JLIRBuilder) = b.blocks[b.insertion]

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

function convert_type_to_jlirtype(builder::JLIRBuilder, a)
    ctx = builder.ctx
    return ccall((:brutus_get_jlirtype, "libbrutus"), 
                 JLIR.Type, 
                 (JLIR.Context, Any), 
                 ctx, a)
end

function convert_value_to_jlirattr(builder::JLIRBuilder, a)
    ctx = builder.ctx
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
