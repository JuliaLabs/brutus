#####
##### Codegen
#####

# This is the Julia interface between Julia's IRCode and JLIR.

function emit_value(b::JLIRBuilder, loc::JLIR.Location, 
        value::GlobalRef, type)
    name = value.name
    v = getproperty(value.mod, value.name)
    return create_constant_op(b, loc, v, type)
end

function emit_value(b::JLIRBuilder, loc::JLIR.Location, 
        value::Core.SSAValue, type)
    @assert(value.id >= 1)
    return getindex(b.values, value.id)
end

function emit_value(b::JLIRBuilder, loc::JLIR.Location, 
        value, type)
    return create_unimplemented_op(b, loc, type)
end

function emit_ftype(b::JLIRBuilder, ret_type)
    argtypes = getfield(ir_code, :argtypes)
    nargs = length(argtypes)
    args = [convert_type_to_jlirtype(b, a) for a in argtypes]
    ret = convert_type_to_jlirtype(b, ret_type)
    return get_functype(b, args, ret)
end

function handle_node!(b::JLIRBuilder, current::Int, 
        v::Vector{JLIR.Value}, stmt::Core.PhiNode, 
        type::Type, loc::JLIR.Location)
    edges = stmt.edges
    values = stmt.values
    found = false
    for (v, e) in zip(edges, values)
        if e == current
            val = emit_value(b, loc, v)
            push!(v, maybe_widen_type(b, loc, val, type))
            found = true
        end
    end
    if !found
        op = create!(b, UndefOp(), loc, convert_type_to_jlirtype(b, type))
        push!(v, JLIR.get_result(op, 0))
    end
end

function walk_cfg_emit_branchargs(b::JLIRBuilder, current::Int, 
        target::Int, loc::JLIR.Location)
    v = JLIR.Value[]
    cfg = get_cfg(b)
    for ind in cfg.blocks[target].stmts
        stmt = get_stmt(b, ind)
        stmt isa Core.PhiNode || break
        type = get_type(b, ind)
        handle_node!(b, v, current, stmt, type, loc)
    end
    return v
end

function emit_op!(b::JLIRBuilder, code::Core.Compiler.IRCode, 
        stmt::Core.GotoIfNot, loc::JLIR.Location, ret::Type)
    label = stmt.label
    v = walk_cfg_emit_branchargs(b, b.insertion, label, loc)
    create!(b, GotoOp(), loc, b.blocks[label], v)
    return true
end

function emit_op!(b::JLIRBuilder, stmt::Core.ReturnNode, 
        loc::JLIR.Location, ret::Type)
    if isdefined(stmt, :val)
        value = maybe_widen_type(b, loc, emit_value(b, loc, stmt.val), ret)
    else
        value = create!(b, UndefOp(), loc)
    end
    create!(b, ReturnOp(), loc, value)
    return true
end

function emit_jlir(ir_code::Core.Compiler.IRCode, ret::Type, name::String)

    # Create builder.
    b = JLIRBuilder(ir_code, name)
    stmts = get_stmts(b)
    types = get_types(b)
    
    # Process.
    #for (ind, (stmt, type)) in enumerate(zip(stmts, types))
    #    lt_ind = location_indices[ind]
    #    loc = lt_ind == 0 ? JLIR.Location() : locations[lt_ind]
    #    is_terminator = false
    #    is_terminator = emit_op!(b, stmt, loc, ret)
    #end
   
    # Create op from state and verify.
    op = finish(b)
    @assert(JLIR.verify(op))
    return op
end
