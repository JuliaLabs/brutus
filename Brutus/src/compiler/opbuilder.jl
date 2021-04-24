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
    reg::JLIR.Region
    code::Core.Compiler.IRCode
    rt::Type
    state::JLIR.OperationState
end

function JLIRBuilder(code::Core.Compiler.IRCode, rt::Type, name::String)
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
    ftype = emit_ftype(ctx, code, rt)
    state = JLIR.create_operation_state("func", locations[1])
    type_attr = JLIR.get_type_attribute(ftype)
    named_type_attr = JLIR.NamedAttribute(ctx, "type", type_attr)
    string_attr = JLIR.get_string_attribute(ctx, name)
    symbol_name_attr = JLIR.NamedAttribute(ctx, "sym_name", string_attr)
    JLIR.push_attributes!(state, named_type_attr)
    JLIR.push_attributes!(state, symbol_name_attr)
    entry_blk, reg = JLIR.add_entry_block!(state, args)
    tr = JLIR.get_first_block(reg)
    nblocks = length(code.cfg.blocks)
    blocks = JLIR.Block[entry_blk]
    for i in 1 : nblocks
        blk = JLIR.Block()
        JLIR.push!(reg, blk)
        push!(blocks, blk)
    end
    return JLIRBuilder(ctx, Ref(2), JLIR.Value[], args, locations, blocks, reg, code, rt, state)
end

set_insertion!(b::JLIRBuilder, blk::Int) = b.insertion[] = blk
get_insertion_block(b::JLIRBuilder) = b.blocks[b.insertion[]]

get_locindices(b::JLIRBuilder) = b.code.stmts.line
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

# Explicitly exposed as part of extern C in codegen.cpp.

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

function convert_jlirtype_to_type(v::JLIR.Type)
    return ccall((:brutus_get_julia_type, "libbrutus"),
                 Type,
                 (JLIR.Type, ),
                 v)
end

function get_functype(ctx::JLIR.Context, args::Vector{JLIR.Type}, ret::JLIR.Type)
    return MLIR.API.mlirFunctionTypeGet(ctx, length(args), args, 1, [ret])
end

function get_functype(ctx::JLIR.Context, args, ret)
    return get_functype(ctx, length(args), map(args) do a
                            convert_type_to_jlirtype(ctx, a)
                        end, 1, [convert_type_to_jlirtype(ctx, ret)])
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
