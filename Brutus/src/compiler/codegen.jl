#####
##### Codegen
#####

# This is the Julia interface between Julia's IRCode and JLIR.

function maybe_widen_type(b::JLIRBuilder, loc::JLIR.Location, 
        value::JLIR.Value, expected_type::Type)
    type = convert_jlirvalue_to_type(value)
    if (type != expected_type && type <: expected_type)
        op = create!(b, PiOp(), loc, value, expected_type)
        return JLIR.get_result(op, 0)
    else
        return value
    end
end

function emit_value(b::JLIRBuilder, loc::JLIR.Location, 
        value, type::Type)
    jlir_type = convert_type_to_jlirtype(b.ctx, type)
    op = create!(b, UnimplementedOp(), loc, jlir_type)
    return JLIR.get_result(op, 0)
end

function emit_value(b::JLIRBuilder, loc::JLIR.Location,
        value::Core.Argument, type::Type)
    idx = value.n
    return b.arguments[idx];
end

function emit_value(b::JLIRBuilder, loc::JLIR.Location, 
        value::Core.SSAValue, type::Type)
    @assert(value.id >= 1)
    return getindex(b.values, value.id)
end

function emit_value(b::JLIRBuilder, loc::JLIR.Location, 
        value::GlobalRef, type::Type)
    name = value.name
    v = getproperty(value.mod, value.name)
    jlir_attr = convert_value_to_jlirattr(b.ctx, v)
    jlir_type = convert_type_to_jlirtype(b.ctx, type)
    op = create!(b, ConstantOp(), loc, jlir_attr, jlir_type)
    return JLIR.get_result(op, 0)
end

function emit_ftype(ctx::JLIR.Context, code::Core.Compiler.IRCode, ret_type::Type)
    argtypes = getfield(code, :argtypes)
    nargs = length(argtypes)
    args = [convert_type_to_jlirtype(ctx, a) for a in argtypes]
    ret = convert_type_to_jlirtype(ctx, ret_type)
    jlir_func_type = get_functype(ctx, args, ret)
    return jlir_func_type
end

function handle_node!(b::JLIRBuilder, current::Int, 
        v::Vector{JLIR.Value}, stmt::Core.PhiNode, 
        type::Type, loc::JLIR.Location)
    edges = stmt.edges
    values = stmt.values
    found = false
    for (v, e) in zip(edges, values)
        if e == current
            val = emit_value(b, loc, v, Any)
            push!(v, maybe_widen_type(b, loc, val, type))
            found = true
        end
    end
    if !found
        jlir_type = convert_type_to_jlirtype(b.ctx, type)
        op = create!(b, UndefOp(), loc, jlir_type)
        push!(v, JLIR.get_result(op, 0))
    end
end

function walk_cfg_emit_branchargs(b::JLIRBuilder, current::Int, 
        target::Int, loc::JLIR.Location)
    v = JLIR.Value[]
    cfg = get_cfg(b)
    for ind in cfg.blocks[target - 1].stmts
        node = get_stmt(b, ind)
        node isa Core.PhiNode || break
        type = get_type(b, ind)
        handle_node!(b, current, v, node, type, loc)
    end
    return v
end

function process_stmt!(b::JLIRBuilder, ind::Int,
        stmt::Nothing, loc::JLIR.Location, type::Type)
    return false
end

function process_stmt!(b::JLIRBuilder, ind::Int,
        stmt, loc::JLIR.Location, type::Type)
    push!(b.values, emit_value(b, loc, stmt, type))
    return false
end

function process_stmt!(b::JLIRBuilder, ind::Int,
        stmt::Core.GotoNode, loc::JLIR.Location, type::Type)
    label = stmt.label
    v = walk_cfg_emit_branchargs(b, b.insertion[], label, loc)
    create!(b, GotoOp(), loc, b.blocks[label], v)
    return true
end

function process_stmt!(b::JLIRBuilder, ind::Int,
        stmt::Core.GotoIfNot, loc::JLIR.Location, type::Type)
    cond = emit_value(b, loc, stmt.cond, Any)
    dest = stmt.dest
    op = create!(b, GotoIfNotOp(), loc, 
                 cond, b.blocks[dest],
                 walk_cfg_emit_branchargs(b, b.insertion[], dest, loc),
                 b.blocks[b.insertion[] + 1],
                 walk_cfg_emit_branchargs(b, b.insertion[], 
                                          b.insertion[] + 1, loc))
    return true
end

function process_stmt!(b::JLIRBuilder, ind::Int,
        stmt::Core.PhiNode, loc::JLIR.Location, type::Type)
    t = convert_type_to_jlirtype(b.ctx, type)
    blk = get_insertion_block(b)
    arg = ccall((:brutusBlockAddArgument, "libbrutus"),
                JLIR.Value,
                (JLIR.Block, JLIR.Type),
                blk, t)
    push!(b.values, arg)
    return false
end

function process_stmt!(b::JLIRBuilder, ind::Int,
        stmt::Core.PiNode, loc::JLIR.Location, type::Type)
    val = stmt.val
    @assert(type == stmt.type)
    jlir_type = convert_type_to_jlirtype(b.ctx, type)
    op = create!(b, PiOp(), loc, emit_value(b, loc, val, Any), jlir_type)
    ctx.values[ind] = JLIR.get_result(op, 0)
    return false
end

function process_stmt!(b::JLIRBuilder, ind::Int,
        stmt::Core.ReturnNode, loc::JLIR.Location, type::Type)
    if isdefined(stmt, :val)
        v = emit_value(b, loc, stmt.val, Any)
        value = maybe_widen_type(b, loc, v, type)
    else
        jlir_type = convert_type_to_jlirtype(b.ctx, type)
        value = create!(b, UndefOp(), loc, jlir_type)
    end
    create!(b, ReturnOp(), loc, value)
    return true
end

function process_stmt!(b::JLIRBuilder, ind::Int, 
        expr::Expr, loc::JLIR.Location, type::Type)
    head = expr.head
    args = expr.args
    jlir_type = convert_type_to_jlirtype(b.ctx, type)
    if head == :invoke
        @assert(args[1] isa Core.MethodInstance)
        mi = args[1]
        callee = emit_value(b, loc, args[2], Any)
        args = JLIR.Value[emit_value(b, loc, a, Any) for a in args[2 : end]]
        op = create!(b, InvokeOp, loc, mi, callee, args, jlir_type)
    elseif head == :call
        callee = emit_value(b, loc, args[1], Any)
        args = JLIR.Value[emit_value(b, loc, a, Any) for a in args]
        op = create!(b, CallOp(), loc, callee, args, jlir_type)
    else
        display(head)
        op = create!(b, UnimplementedOp(), loc, jlir_type)
    end
    res = JLIR.get_result(op, 0)
    push!(b.values, res)
    return false
end

function emit_jlir(ir_code::Core.Compiler.IRCode, rt::Type, name::String)
    # Create builder.
    b = JLIRBuilder(ir_code, rt, name)

    # Create branch from entry block.
    v = walk_cfg_emit_branchargs(b, 1, 2, b.locations[1])
    goto = create_goto_op(JLIR.Location(b.ctx), b.blocks[2], v)
    push!(b.blocks[1], goto)

    # Process.
    location_indices = get_locindices(b)
    stmts = get_stmts(b)
    types = get_types(b)
    for (ind, (stmt, type)) in enumerate(zip(stmts, types))
        lt_ind = location_indices[ind]
        loc = lt_ind == 0 ? JLIR.Location() : b.locations[lt_ind]
        is_terminator = false
        is_terminator = process_stmt!(b, ind, stmt, loc, type)
        if is_terminator
            b.insertion[] += 1
        end
    end

    # Create op from state and verify.
    op = finish(b)
    JLIR.verify(op)
    JLIR.dump(op)
    return op
end
