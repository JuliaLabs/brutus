#ifndef JL_DIALECT_JLIR_H
#define JL_DIALECT_JLIR_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "julia.h"
#include "brutus/brutus_internal.h"

namespace mlir {
namespace jlir {

/// This is the definition of the Julia dialect. A dialect inherits from
/// mlir::Dialect and registers custom attributes, operations, and types (in its
/// constructor). It can also override some general behavior exposed via virtual
/// methods.
class JLIRDialect : public mlir::Dialect {
public:
    explicit JLIRDialect(mlir::MLIRContext *ctx);

    // /// A hook used to materialize constant values with the given type.
    // Operation *materializeConstant(OpBuilder &builder, Attribute value, Type type,
    //                                Location loc) override;

    // /// Parse an instance of a type registered to the toy dialect.
    // mlir::Type parseType(mlir::DialectAsmParser &parser) const override;

    /// Print an instance of a type registered to the toy dialect.
    void printType(mlir::Type type,
                   mlir::DialectAsmPrinter &printer) const override;

    void printAttribute(mlir::Attribute attr,
                        mlir::DialectAsmPrinter &printer) const override;

    /// Provide a utility accessor to the dialect namespace. This is used by
    /// several utilities for casting between dialects.
    static llvm::StringRef getDialectNamespace() { return "jlir"; }

    static std::string showValue(jl_value_t *value);
};

/// JLIR Types

class JuliaTypeStorage : public mlir::TypeStorage {
public:
    JuliaTypeStorage(jl_datatype_t *datatype) : datatype(datatype) {}

    using KeyTy = jl_datatype_t*;

    bool operator==(const KeyTy &key) const {
        return jl_egal((jl_value_t*)key, (jl_value_t*)datatype);
    }

    static llvm::hash_code hashKey(const KeyTy &key) {
        return jl_object_id((jl_value_t*)key);
    }

    static JuliaTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                       const KeyTy &key) {
        return new (allocator.allocate<JuliaTypeStorage>())
            JuliaTypeStorage(key);
    }

    jl_datatype_t *datatype;
};

class JuliaType
    : public mlir::Type::TypeBase<JuliaType, mlir::Type, JuliaTypeStorage> {
public:
    using Base::Base;

    static bool kindof(unsigned kind) { return kind == JLIRTypes::JuliaType; }

    static JuliaType get(mlir::MLIRContext *context, jl_datatype_t *datatype) {
        // unwrap Core.Compiler.Const
        if (jl_isa((jl_value_t*)datatype, const_type)) {
            datatype = (jl_datatype_t*)jl_typeof(
                jl_get_field((jl_value_t*)datatype, "val"));
        }

        return Base::get(context, JLIRTypes::JuliaType, datatype);
    }

    jl_datatype_t *getDatatype() {
        // `getImpl` returns a pointer to the internal storage instance
        return getImpl()->datatype;
    }
};

class JuliaValueAttrStorage : public mlir::AttributeStorage {
public:
    JuliaValueAttrStorage(jl_value_t *value) : value(value) {}

    using KeyTy = jl_value_t*;

    bool operator==(const KeyTy &key) const {
        return jl_egal(key, value);
    }

    static llvm::hash_code hashKey(const KeyTy &key) {
        return jl_object_id(key);
    }

    static JuliaValueAttrStorage *construct(
        mlir::AttributeStorageAllocator &allocator, const KeyTy &key) {
        return new (allocator.allocate<JuliaValueAttrStorage>())
            JuliaValueAttrStorage(key);
    }

    jl_value_t *value;
};

class JuliaValueAttr : public mlir::Attribute::AttrBase<
        JuliaValueAttr, mlir::Attribute, JuliaValueAttrStorage> {
public:
    using Base::Base;

    static JuliaValueAttr get(mlir::MLIRContext *context, jl_value_t *value) {
        return Base::get(context, value);
    }

    jl_value_t *getValue() {
        return getImpl()->value;
    }
};

} // end namespace jlir
} // end namespace mlir

/// Include the auto-generated header file containing the declarations of the
/// JuliaIR operations.
#define GET_OP_CLASSES
#include "brutus/Dialect/Julia/JuliaOps.h.inc"


#endif // JL_DIALECT_JLIR_H
