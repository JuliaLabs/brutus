function create_unimplemented_op(loc::JLIR.Location, type)
    state = JLIR.create_operation_state("jlir::unimplemented", loc)
    JLIR.push_results!(state, 1, type)
    return JLIR.Operation(state)
end

function create_constant_op(loc::JLIR.Location, value, type)
    state = JLIR.create_operation_state("jlir::constant", loc)
    JLIR.push_operands!(state, 1, value)
    JLIR.push_results!(state, 1, type)
    return JLIR.Operation(state)
end

function create_call_op(loc::JLIR.Location, callee, arguments, type)
    state = JLIR.create_operation_state("jlir::call", loc)
    operands = [callee, arguments...]
    JLIR.push_operands!(state, length(operands), operands)
    JLIR.push_results!(state, 1, type)
    return JLIR.Operation(state)
end
