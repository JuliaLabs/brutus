
#include "brutus/Dialect/Julia/JuliaDialect.h"
#include "brutus/Dialect/Julia/JuliaOps.h"

using namespace mlir;
using namespace mlir::jlir;

#include "brutus/Dialect/Julia/JuliaOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Standalone dialect.
//===----------------------------------------------------------------------===//

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
void JLIRDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "brutus/Dialect/Julia/JuliaOps.cpp.inc"
        >();
    addTypes<JuliaType>();
    addAttributes<
        JuliaValueAttr
        >();
}

// void JLIRDialect::printType(mlir::Type type,
//                             mlir::DialectAsmPrinter &printer) const {
//     assert(type.isa<JuliaType>());
//     printer << showValue((jl_value_t*)type.cast<JuliaType>().getDatatype());
// }

// void JLIRDialect::printAttribute(mlir::Attribute attr,
//                                  mlir::DialectAsmPrinter &printer) const {
//     // NOTE: printing values may use illegal characters (such as quotes?)
//     assert(attr.isa<JuliaValueAttr>());
//     printer << showValue(attr.cast<JuliaValueAttr>().getValue());
// }

// std::string JLIRDialect::showValue(jl_value_t *value) {
//     ios_t str_;
//     ios_mem(&str_, 10);
//     JL_STREAM *str = (JL_STREAM*)&str_;
//     jl_static_show(str, value);
//     str_.buf[str_.size] = '\0';
//     std::string s = str_.buf;
//     ios_close(&str_);
//     return s;
// }

