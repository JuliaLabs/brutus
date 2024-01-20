#ifndef JULIA_DIALECT
#define JULIA_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def JLIR_Dialect : Dialect {
    let name = "jlir";
    let summary = "";
    let description = [{
    }];
    let cppNamespace = "::mlir::jlir";

  let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;
}

//===----------------------------------------------------------------------===//
// Dialect types                                                              //
//===----------------------------------------------------------------------===//


def JLIR_JuliaType : DialectType<JLIR_Dialect, CPred<"$_self.isa<JuliaType>()">, "JLIR Julia type">;

def JLIR_JuliaValueAttr : Attr<CPred<"$_self.isa<JuliaValueAttr>()">,
                               "JLIR Julia Value attribute"> {
    let storageType = [{ JuliaValueAttr }];
    let returnType = [{ jl_value_t * }];
}


// Base class for JLIR dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class JLIR_Op<string mnemonic, list<Trait> traits = []> :
    Op<JLIR_Dialect, mnemonic, traits>;

// Base clase for JLIR intrinsics
class JLIR_IntrinsicBuiltinOp<string mnemonic, list<Trait> traits = []> :
    JLIR_Op<mnemonic, traits> {

    let arguments = (ins Variadic<JLIR_JuliaType>:$arguments);
    let results = (outs JLIR_JuliaType);
}

class JLIR_ArithmeticOp<string mnemonic, list<Trait> traits = []>:
    JLIR_IntrinsicBuiltinOp<mnemonic, !listconcat([NoSideEffect, SameOperandsAndResultType], traits)> {

    let arguments = (ins JLIR_JuliaType:$rhs, JLIR_JuliaType:$lhs);
    let results = (outs JLIR_JuliaType);
}

class JLIR_UnaryArithmeticOp<string mnemonic, list<Trait> traits = []>:
    JLIR_IntrinsicBuiltinOp<mnemonic, !listconcat([NoSideEffect, SameOperandsAndResultType], traits)> {

    let arguments = (ins JLIR_JuliaType:$arg);
    let results = (outs JLIR_JuliaType);
}

#endif // STANDALONE_DIALECT
