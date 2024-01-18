module jlir

import ...IR: NamedAttribute, MLIRType, Value, Location, Block, Region, Attribute, create_operation, context, IndexType
import ..Dialects: namedattribute, operandsegmentsizes
import ...API


"""
`arraytomemref`

TODO
"""
function arraytomemref(a::Value; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[a, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.arraytomemref", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`call`

TODO
"""
function call(callee::Value, arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[callee, arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.call", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`constant`

TODO
"""
function constant(; result_0::MLIRType, value, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("value", value), ]
    
    create_operation(
        "jlir.constant", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`convertstd`

"""
function convertstd(input::Value; output::MLIRType, location=Location())
    results = MLIRType[output, ]
    operands = Value[input, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.convertstd", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`gotoifnot`

TODO
"""
function gotoifnot(condition::Value, branchOperands::Vector{Value}, fallthroughOperands::Vector{Value}; branchDest::Block, fallthroughDest::Block, location=Location())
    results = MLIRType[]
    operands = Value[condition, branchOperands..., fallthroughOperands..., ]
    owned_regions = Region[]
    successors = Block[branchDest, fallthroughDest, ]
    attributes = NamedAttribute[]
    push!(attributes, operandsegmentsizes([1, length(branchOperands), length(fallthroughOperands), ]))
    
    create_operation(
        "jlir.gotoifnot", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`goto`

TODO
"""
function goto(operands::Vector{Value}; dest::Block, location=Location())
    results = MLIRType[]
    operands = Value[operands..., ]
    owned_regions = Region[]
    successors = Block[dest, ]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.goto", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`invoke`

TODO
"""
function invoke(callee::Value, arguments::Vector{Value}; result_0::MLIRType, methodInstance, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[callee, arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[namedattribute("methodInstance", methodInstance), ]
    
    create_operation(
        "jlir.invoke", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`_apply`

"""
function _apply(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir._apply", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`_apply_iterate`

"""
function _apply_iterate(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir._apply_iterate", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`_apply_latest`

"""
function _apply_latest(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir._apply_latest", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`_apply_pure`

"""
function _apply_pure(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir._apply_pure", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`_expr`

"""
function _expr(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir._expr", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`_typevar`

"""
function _typevar(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir._typevar", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`applicable`

"""
function applicable(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.applicable", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`apply_type`

"""
function apply_type(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.apply_type", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`arrayref`

"""
function arrayref(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.arrayref", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`arrayset`

"""
function arrayset(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.arrayset", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`arraysize`

"""
function arraysize(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.arraysize", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`const_arrayref`

"""
function const_arrayref(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.const_arrayref", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`fieldtype`

"""
function fieldtype(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.fieldtype", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`getfield`

"""
function getfield(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.getfield", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`ifelse`

"""
function ifelse(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.ifelse", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`===`

"""
function ===(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.===", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`isa`

"""
function isa(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.isa", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`isdefined`

"""
function isdefined(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.isdefined", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`issubtype`

"""
function issubtype(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.issubtype", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`nfields`

"""
function nfields(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.nfields", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`setfield!`

"""
function setfield!(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.setfield!", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`sizeof`

"""
function sizeof(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.sizeof", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`svec`

"""
function svec(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.svec", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`throw`

"""
function throw(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.throw", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`tuple`

"""
function tuple(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.tuple", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`typeassert`

"""
function typeassert(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.typeassert", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`typeof`

"""
function typeof(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.typeof", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`abs_float`

"""
function abs_float(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.abs_float", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`add_float`

"""
function add_float(rhs::Value, lhs::Value; result_0=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[rhs, lhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result_0 != nothing) && push!(results, result_0)
    
    create_operation(
        "jlir.add_float", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`add_int`

"""
function add_int(rhs::Value, lhs::Value; result_0=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[rhs, lhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result_0 != nothing) && push!(results, result_0)
    
    create_operation(
        "jlir.add_int", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`add_ptr`

"""
function add_ptr(rhs::Value, lhs::Value; result_0=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[rhs, lhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result_0 != nothing) && push!(results, result_0)
    
    create_operation(
        "jlir.add_ptr", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`and_int`

"""
function and_int(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.and_int", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`arraylen`

"""
function arraylen(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.arraylen", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`ashr_int`

"""
function ashr_int(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.ashr_int", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`bitcast`

"""
function bitcast(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.bitcast", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`bswap_int`

"""
function bswap_int(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.bswap_int", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`ceil_llvm`

"""
function ceil_llvm(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.ceil_llvm", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`cglobal`

"""
function cglobal(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.cglobal", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`checked_sadd_int`

"""
function checked_sadd_int(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.checked_sadd_int", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`checked_sdiv_int`

"""
function checked_sdiv_int(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.checked_sdiv_int", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`checked_smul_int`

"""
function checked_smul_int(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.checked_smul_int", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`checked_srem_int`

"""
function checked_srem_int(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.checked_srem_int", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`checked_ssub_int`

"""
function checked_ssub_int(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.checked_ssub_int", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`checked_uadd_int`

"""
function checked_uadd_int(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.checked_uadd_int", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`checked_udiv_int`

"""
function checked_udiv_int(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.checked_udiv_int", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`checked_umul_int`

"""
function checked_umul_int(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.checked_umul_int", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`checked_urem_int`

"""
function checked_urem_int(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.checked_urem_int", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`checked_usub_int`

"""
function checked_usub_int(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.checked_usub_int", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`copysign_float`

"""
function copysign_float(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.copysign_float", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`ctlz_int`

"""
function ctlz_int(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.ctlz_int", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`ctpop_int`

"""
function ctpop_int(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.ctpop_int", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`cttz_int`

"""
function cttz_int(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.cttz_int", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`div_float`

"""
function div_float(rhs::Value, lhs::Value; result_0=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[rhs, lhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result_0 != nothing) && push!(results, result_0)
    
    create_operation(
        "jlir.div_float", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`eq_float`

"""
function eq_float(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.eq_float", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`eq_int`

"""
function eq_int(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.eq_int", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`flipsign_int`

"""
function flipsign_int(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.flipsign_int", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`floor_llvm`

"""
function floor_llvm(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.floor_llvm", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`fma_float`

"""
function fma_float(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.fma_float", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`fpext`

"""
function fpext(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.fpext", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`fpiseq`

"""
function fpiseq(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.fpiseq", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`fpislt`

"""
function fpislt(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.fpislt", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`fptosi`

"""
function fptosi(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.fptosi", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`fptoui`

"""
function fptoui(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.fptoui", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`fptrunc`

"""
function fptrunc(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.fptrunc", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`le_float`

"""
function le_float(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.le_float", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`lshr_int`

"""
function lshr_int(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.lshr_int", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`lt_float`

"""
function lt_float(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.lt_float", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`mul_float`

"""
function mul_float(rhs::Value, lhs::Value; result_0=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[rhs, lhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result_0 != nothing) && push!(results, result_0)
    
    create_operation(
        "jlir.mul_float", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`mul_int`

"""
function mul_int(rhs::Value, lhs::Value; result_0=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[rhs, lhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result_0 != nothing) && push!(results, result_0)
    
    create_operation(
        "jlir.mul_int", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`muladd_float`

"""
function muladd_float(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.muladd_float", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`ne_float`

"""
function ne_float(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.ne_float", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`ne_int`

"""
function ne_int(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.ne_int", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`neg_float`

"""
function neg_float(arg::Value; result_0=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result_0 != nothing) && push!(results, result_0)
    
    create_operation(
        "jlir.neg_float", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`neg_int`

"""
function neg_int(arg::Value; result_0=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[arg, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result_0 != nothing) && push!(results, result_0)
    
    create_operation(
        "jlir.neg_int", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`not_int`

"""
function not_int(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.not_int", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`or_int`

"""
function or_int(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.or_int", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`pointerref`

"""
function pointerref(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.pointerref", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`pointerset`

"""
function pointerset(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.pointerset", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`rem_float`

"""
function rem_float(rhs::Value, lhs::Value; result_0=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[rhs, lhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result_0 != nothing) && push!(results, result_0)
    
    create_operation(
        "jlir.rem_float", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`rint_llvm`

"""
function rint_llvm(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.rint_llvm", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`sdiv_int`

"""
function sdiv_int(rhs::Value, lhs::Value; result_0=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[rhs, lhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result_0 != nothing) && push!(results, result_0)
    
    create_operation(
        "jlir.sdiv_int", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`sext_int`

"""
function sext_int(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.sext_int", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`shl_int`

"""
function shl_int(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.shl_int", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`sitofp`

"""
function sitofp(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.sitofp", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`sle_int`

"""
function sle_int(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.sle_int", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`slt_int`

"""
function slt_int(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.slt_int", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`sqrt_llvm`

"""
function sqrt_llvm(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.sqrt_llvm", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`sqrt_llvm_fast`

"""
function sqrt_llvm_fast(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.sqrt_llvm_fast", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`srem_int`

"""
function srem_int(rhs::Value, lhs::Value; result_0=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[rhs, lhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result_0 != nothing) && push!(results, result_0)
    
    create_operation(
        "jlir.srem_int", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`sub_float`

"""
function sub_float(rhs::Value, lhs::Value; result_0=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[rhs, lhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result_0 != nothing) && push!(results, result_0)
    
    create_operation(
        "jlir.sub_float", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`sub_int`

"""
function sub_int(rhs::Value, lhs::Value; result_0=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[rhs, lhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result_0 != nothing) && push!(results, result_0)
    
    create_operation(
        "jlir.sub_int", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`sub_ptr`

"""
function sub_ptr(rhs::Value, lhs::Value; result_0=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[rhs, lhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result_0 != nothing) && push!(results, result_0)
    
    create_operation(
        "jlir.sub_ptr", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`trunc_int`

"""
function trunc_int(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.trunc_int", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`trunc_llvm`

"""
function trunc_llvm(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.trunc_llvm", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`udiv_int`

"""
function udiv_int(rhs::Value, lhs::Value; result_0=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[rhs, lhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result_0 != nothing) && push!(results, result_0)
    
    create_operation(
        "jlir.udiv_int", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`uitofp`

"""
function uitofp(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.uitofp", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`ule_int`

"""
function ule_int(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.ule_int", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`ult_int`

"""
function ult_int(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.ult_int", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`urem_int`

"""
function urem_int(rhs::Value, lhs::Value; result_0=nothing::Union{Nothing, MLIRType}, location=Location())
    results = MLIRType[]
    operands = Value[rhs, lhs, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    (result_0 != nothing) && push!(results, result_0)
    
    create_operation(
        "jlir.urem_int", location;
        operands, owned_regions, successors, attributes,
        results=(length(results) == 0 ? nothing : results),
        result_inference=(length(results) == 0 ? true : false)
    )
end

"""
`xor_int`

"""
function xor_int(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.xor_int", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`zext_int`

"""
function zext_int(arguments::Vector{Value}; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[arguments..., ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.zext_int", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`pi`

TODO
"""
function pi(input::Value; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[input, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.pi", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`return_`

The \"return\" operation represents a return operation within a function.
The operand type must match the signature of the function that contains
the operation. For example:

```mlir
func @foo() -> i32 {
    ...
    jlir.return %0 : i32
}
```
"""
function return_(input::Value; location=Location())
    results = MLIRType[]
    operands = Value[input, ]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.return", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`undef`

TODO
"""
function undef(; result_0::MLIRType, location=Location())
    results = MLIRType[result_0, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.undef", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

"""
`unimplemented`

unimplemented
"""
function unimplemented(; type::MLIRType, location=Location())
    results = MLIRType[type, ]
    operands = Value[]
    owned_regions = Region[]
    successors = Block[]
    attributes = NamedAttribute[]
    
    create_operation(
        "jlir.unimplemented", location;
        operands, owned_regions, successors, attributes,
        results=results,
        result_inference=false
    )
end

end # jlir
