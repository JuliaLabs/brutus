#include "brutus/Dialect/Julia/JuliaOps.h"

#include "julia.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"

using namespace mlir;
using namespace mlir::jlir;

void UnimplementedOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, jl_datatype_t *type) {
    state.addTypes(JuliaType::get(builder.getContext(), type));
}

void UndefOp::build(mlir::OpBuilder &builder, mlir::OperationState &state) {
    state.addTypes(JuliaType::get(builder.getContext(), jl_any_type));
}

void ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, jl_value_t *value, jl_datatype_t *type) {
    state.addAttribute("value", JuliaValueAttr::get(builder.getContext(), value));
    state.addTypes(JuliaType::get(builder.getContext(), type));
}

mlir::OpFoldResult ConstantOp::fold(llvm::ArrayRef<mlir::Attribute> operands) {
    return getValueAttr();
}

void PiOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                 Value value, jl_datatype_t *type) {
    state.addTypes(JuliaType::get(builder.getContext(), type));
    state.addOperands(value);
}

//===----------------------------------------------------------------------===//
// CallOp

void CallOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   mlir::Value callee, ArrayRef<mlir::Value> arguments, jl_datatype_t *type) {
    state.addTypes(JuliaType::get(builder.getContext(), type));
    state.addOperands(callee);
    state.addOperands(arguments);
}

//===----------------------------------------------------------------------===//
// InvokeOp

void InvokeOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                     jl_method_instance_t *methodInstance, mlir::Value callee,
                     ArrayRef<mlir::Value> arguments, jl_datatype_t *type) {
    state.addTypes(JuliaType::get(builder.getContext(), type));
    state.addOperands(callee);
    state.addOperands(arguments);
    state.addAttribute("methodInstance",
                       JuliaValueAttr::get(builder.getContext(), (jl_value_t*)methodInstance));
}

//===----------------------------------------------------------------------===//
// ReturnOp

static mlir::LogicalResult verify(ReturnOp op) {
    // We know that the parent operation is a function, because of the 'HasParent'
    // trait attached to the operation definition.
    auto function = cast<func::FuncOp>(op->getParentOp());

    // const auto &results = function.getType().getResults();
    // if (results.size() != 1)
    //     return function.emitOpError() << "does not return exactly one value";

    // // check that result type of function matches the operand type
    // if (results.front() != op.getOperand().getType())
    //     return op.emitError() << "type of return operand ("
    //                           << op.getOperand().getType()
    //                           << ") doesn't match function result type ("
    //                           << results.front() << ")";

    return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "brutus/Dialect/Julia/JuliaOps.cpp.inc"
