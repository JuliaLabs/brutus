#include "brutus/Dialect/Julia/JuliaOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/StandardTypes.h"

using namespace mlir;
using namespace mlir::jlir;

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
JLIRDialect::JLIRDialect(mlir::MLIRContext *ctx) : mlir::Dialect("jlir", ctx) {
    addOperations<
#define GET_OP_LIST
#include "brutus/Dialect/Julia/JuliaOps.cpp.inc"
        >();
    addTypes<JuliaType>();
    addAttributes<JuliaValueAttr>();
}

void JLIRDialect::printType(mlir::Type type,
                            mlir::DialectAsmPrinter &printer) const {
    assert(type.isa<JuliaType>());
    printer << showValue((jl_value_t*)type.cast<JuliaType>().getDatatype());
}

void JLIRDialect::printAttribute(mlir::Attribute attr,
                                 mlir::DialectAsmPrinter &printer) const {
    // NOTE: printing values may use illegal characters (such as quotes?)
    assert(attr.isa<JuliaValueAttr>());
    printer << showValue(attr.cast<JuliaValueAttr>().getValue());
}

std::string JLIRDialect::showValue(jl_value_t *value) {
    ios_t str_;
    ios_mem(&str_, 10);
    JL_STREAM *str = (JL_STREAM*)&str_;
    jl_static_show(str, value);
    str_.buf[str_.size] = '\0';
    std::string s = str_.buf;
    ios_close(&str_);
    return s;
}

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
    return valueAttr();
}

void PiOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                 Value value, jl_datatype_t *type) {
    state.addTypes(JuliaType::get(builder.getContext(), type));
    state.addOperands(value);
}

//===----------------------------------------------------------------------===//
// CallOp

void CallOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   jl_datatype_t *type, mlir::Value callee, ArrayRef<mlir::Value> arguments) {
    state.addTypes(JuliaType::get(builder.getContext(), type));
    state.addOperands(callee);
    state.addOperands(arguments);
}

//===----------------------------------------------------------------------===//
// InvokeOp

void InvokeOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                     jl_method_instance_t *methodInstance,
                     ArrayRef<mlir::Value> arguments) {
    state.addTypes(JuliaType::get(builder.getContext(), jl_any_type));
    state.addOperands(arguments);
    state.addAttribute("methodInstance",
                       JuliaValueAttr::get(builder.getContext(), (jl_value_t*)methodInstance));
}

//===----------------------------------------------------------------------===//
// ReturnOp

static mlir::LogicalResult verify(ReturnOp op) {
    // We know that the parent operation is a function, because of the 'HasParent'
    // trait attached to the operation definition.
    auto function = cast<FuncOp>(op.getParentOp());

    const auto &results = function.getType().getResults();
    if (results.size() != 1)
        return function.emitOpError() << "does not return exactly one value";

    // check that result type of function matches the operand type
    if (results.front() != op.getOperand().getType())
        return op.emitError() << "type of return operand ("
                              << op.getOperand().getType()
                              << ") doesn't match function result type ("
                              << results.front() << ")";

    return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "brutus/Dialect/Julia/JuliaOps.cpp.inc"
