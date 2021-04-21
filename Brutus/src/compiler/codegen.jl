#####
##### Codegen
#####

# This is the Julia interface between Julia's IRCode and JLIR.

function emit_value(builder::JLIRBuilder, loc::JLIR.Location, value::GlobalRef, type)
    name = value.name
    v = getproperty(value.mod, value.name)
    return create_constant_op(builder, loc, v, type)
end

function emit_value(builder::JLIRBuilder, loc::JLIR.Location, value::Core.SSAValue, type)
    @assert(value.id >= 1)
    return getindex(builder.values, value.id)
end

function emit_value(builder::JLIRBuilder, loc::JLIR.Location, value, type)
    return create_unimplemented_op(builder, loc, type)
end

function emit_ftype(builder::JLIRBuilder, ir_code::Core.Compiler.IRCode, ret_type)
    argtypes = getfield(ir_code, :argtypes)
    nargs = length(argtypes)
    args = [convert_type_to_mlir(builder, a) for a in argtypes]
    ret = convert_type_to_mlir(builder, ret_type)
    return get_functype(builder, args, ret)
end

function process_node!(b::JLIRBuilder)
end

function walk_cfg_emit_branchargs(builder, 
        cfg::Core.Compiler.CFG, current_block::Int, 
        target_block::Int, stmts, types, loc::JLIR.Location)
    v = JLIR.Value[]
    for stmt in cfg.blocks[target].stmts
        handle_node!(builder, v, stmt, loc)
    end
    return v
end

function emit_jlir(builder::JLIRBuilder, 
        ir_code::Core.Compiler.IRCode, ret::Type, name::String)

    # Setup.
    irstream = ir_code.stmts
    location_indices = getfield(irstream, :line)
    linetable = getfield(ir_code, :linetable)
    locations = extract_linetable_meta(builder, linetable)
    argtypes = getfield(ir_code, :argtypes)
    args = [convert_type_to_mlir(builder, a) for a in argtypes]
    state = JLIR.create_operation_state(name, locations[1])
    entry_blk, reg = JLIR.add_entry_block!(state, args)
    cfg = ir_code.cfg
    cfg_blocks = cfg.blocks
    nblocks = length(cfg_blocks)
    blocks = JLIR.Block[JLIR.push_new_block!(reg) for _ in 1 : nblocks]
    pushfirst!(blocks, entry_blk)
    builder.blocks = blocks
    stmts = irstream.inst
    types = irstream.type
    v = walk_cfg_emit_branchargs(builder, cfg, 1, 2, stmts, types, locations[0])
    goto = create_goto_op(Location(builder.ctx), blocks[2], v)
    push!(builder, goto)
    set_insertion!(builder, 2)

    # Process.
    for (ind, (stmt, type)) in enumerate(zip(stmts, types))
        loc = linetable[ind] == 0 ? JLIR.Location() : locations[ind]
        is_terminator = false
        process_node!(builder, stmt, loc)
    end
   
    # Create op from state and verify.
    op = JLIR.Operation(state)
    @assert(JLIR.verify(op))
    return op
end

function emit_jlir(ir_code::Core.Compiler.IRCode, ret::Type, name::String)
    b = JLIRBuilder()
    return create_func_op(b, ir_code, ret, name)
end
