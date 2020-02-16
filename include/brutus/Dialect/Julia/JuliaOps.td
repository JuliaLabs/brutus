//===----------------------------------------------------------------------===//
//
// Defines the operations of the Julia dialect.
//
//===----------------------------------------------------------------------===//

#ifndef JULIA_MLIR_JLIR_TD
#define JULIA_MLIR_JLIR_TD

include "mlir/IR/OpBase.td"

// Provide a definition of the 'JLIR' dialect in the ODS framework so that we
// can define our operations.
def JLIR_Dialect : Dialect {
    let name = "jlir";
    let cppNamespace = "jlir";
}

def JLIR_JuliaType : Type<CPred<"$_self.isa<JuliaType>()">, "JLIR Julia type">;

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
class JLIR_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<JLIR_Dialect, mnemonic, traits>;

// Base clase for JLIR intrinsics
class JLIR_IntrinsicOp<string mnemonic, list<OpTrait> traits = []> :
    JLIR_Op<mnemonic, traits> {

    let arguments = (ins Variadic<JLIR_JuliaType>:$arguments);
    let results = (outs JLIR_JuliaType);

    let builders = [
        OpBuilder<"Builder *builder, OperationState &state,"
                  "jl_datatype_t *type, ArrayRef<Value> arguments">
    ];
}

class JLIR_ArithmeticOp<string mnemonic, list<OpTrait> traits = []>:
    JLIR_IntrinsicOp<mnemonic, !listconcat([NoSideEffect, SameOperandsAndResultType], traits)> {

    let arguments = (ins JLIR_JuliaType:$rhs, JLIR_JuliaType:$lhs);
    let results = (outs JLIR_JuliaType);
}

class JLIR_UnaryArithmeticOp<string mnemonic, list<OpTrait> traits = []>:
    JLIR_IntrinsicOp<mnemonic, !listconcat([NoSideEffect, SameOperandsAndResultType], traits)> {

    let arguments = (ins JLIR_JuliaType:$arg);
    let results = (outs JLIR_JuliaType);
}

//===----------------------------------------------------------------------===//
// JLIR Operations
//===----------------------------------------------------------------------===//

// We define a Julia operation by inheriting from our base 'Julia_Op' class above.
// Here we provide the mnemonic and a list of traits for the operation.

def UnimplementedOp : JLIR_Op<"unimplemented"> {
    let summary = "unimplemented operation";
    let description = [{
        unimplemented
    }];

    let results = (outs JLIR_JuliaType);
    let builders = [
        OpBuilder<"Builder *builder, OperationState &state,"
                  "jl_datatype_t *type = jl_any_type">
    ];
}

def UndefOp : JLIR_Op<"undef"> {
    let summary = "undef generating operation";
    let description = [{
        TODO
    }];

    let results = (outs JLIR_JuliaType);
    let builders = [
        OpBuilder<"Builder *builder, OperationState &state">
    ];
}

def ConstantOp : JLIR_Op<"constant", [NoSideEffect]> {
    let summary = "constant generating operation";
    let description = [{
        TODO
    }];

    let arguments = (ins JLIR_JuliaValueAttr:$value);
    let results = (outs JLIR_JuliaType);
    let builders = [
        OpBuilder<"Builder *builder, OperationState &state,"
                  "jl_value_t *value, jl_datatype_t *type = jl_any_type">
    ];

    // Set the folder bit so that we can implement constant folders.
    let hasFolder = 1;
}

def CallOp : JLIR_Op<"call"> {
    let summary = "generic call operation";
    let description = [{
        TODO
    }];

    // The generic call operation takes a symbol reference attribute as the
    // callee, and inputs for the call.
    let arguments = (ins JLIR_JuliaType:$callee, Variadic<JLIR_JuliaType>:$arguments);

    // The generic call operation returns a single value of TensorType.
    let results = (outs JLIR_JuliaType);

    // Add custom build methods for the generic call operation.
    let builders = [
        OpBuilder<"Builder *builder, OperationState &state,"
                  "jl_datatype_t *type, Value callee, ArrayRef<Value> arguments">
    ];

    // let hasCanonicalizer = 1;
}

def InvokeOp : JLIR_Op<"invoke"> {
    let summary = "invoke operation";
    let description = [{
        TODO
    }];

    let arguments = (ins JLIR_JuliaValueAttr:$methodInstance, Variadic<JLIR_JuliaType>:$arguments);

    let results = (outs JLIR_JuliaType);

    let builders = [
        OpBuilder<"Builder *builder, OperationState &state,"
                  "jl_method_instance_t *methodInstance,"
                  "ArrayRef<Value> arguments">
    ];
}

def GotoOp : JLIR_Op<"goto", [Terminator]> {
    let summary = "goto operation";
    let description = [{
        TODO
    }];

    let arguments = (ins Variadic<JLIR_JuliaType>:$operands);

    let builders = [
        OpBuilder<"Builder *builder, OperationState &state, Block *dest,"
                  "ValueRange operands = {}",
                  [{ state.addSuccessor(dest, operands); }]>
    ];
}

def GotoIfNotOp : JLIR_Op<"gotoifnot", [Terminator]> {
    let summary = "goto if not operation";
    let description = [{
        TODO
    }];

    let arguments = (ins JLIR_JuliaType:$condition, Variadic<JLIR_JuliaType>:$operands);

    let builders = [
        OpBuilder<"Builder *builder, OperationState &state, Value condition,"
                  "Block *branchDest, ValueRange branchOperands,"
                  "Block *fallthroughDest, ValueRange fallthroughOperands",
                  [{
                      state.addOperands(condition);
                      state.addSuccessor(branchDest, branchOperands);
                      state.addSuccessor(fallthroughDest, fallthroughOperands);
                  }]>
    ];
}

def ReturnOp : JLIR_Op<"return", [Terminator, HasParent<"FuncOp">]> {
    let summary = "return operation";
    let description = [{
        The "return" operation represents a return operation within a function.
        The operand type must match the signature of the function that contains
        the operation. For example:

        ```mlir
        func @foo() -> i32 {
            ...
            jlir.return %0 : i32
        }
        ```
    }];

    // The return operation takes an optional input operand to return. This
    // value must match the return type of the enclosing function.
    let arguments = (ins Variadic<JLIR_JuliaType>:$input);

    // Allow building a ReturnOp with no return operand.
    let builders = [OpBuilder<
        "Builder *b, OperationState &state", [{ build(b, state, llvm::None); }]
    >];

    // Provide extra utility definitions on the c++ operation class definition.
    let extraClassDeclaration = [{
        bool hasOperand() { return getNumOperands() != 0; }
    }];

    // Invoke a static verify method to verify this return operation.
    let verifier = [{ return ::verify(*this); }];
}

def PiOp : JLIR_Op<"pi", [NoSideEffect]> {
    let summary = "pi operation";
    let description = [{
        TODO
    }];

    let arguments = (ins JLIR_JuliaType:$input);
    let results = (outs JLIR_JuliaType);
    let builders = [
        OpBuilder<"Builder *builder, OperationState &state,"
                  "Value value, jl_datatype_t *type">
    ];
}

/*  wrap and unwrap */
def JLIR_bitcast : JLIR_IntrinsicOp<"bitcast", [NoSideEffect]>;

/*  arithmetic */
def JLIR_neg_int   : JLIR_UnaryArithmeticOp<"neg_int">;
def JLIR_add_int   : JLIR_ArithmeticOp<"add_int", [Commutative]>;
def JLIR_sub_int   : JLIR_ArithmeticOp<"sub_int">;
def JLIR_mul_int   : JLIR_ArithmeticOp<"mul_int", [Commutative]>;
def JLIR_sdiv_int  : JLIR_ArithmeticOp<"sdiv_int">;
def JLIR_udiv_int  : JLIR_ArithmeticOp<"udiv_int">;
def JLIR_srem_int  : JLIR_ArithmeticOp<"srem_int">;
def JLIR_urem_int  : JLIR_ArithmeticOp<"urem_int">;
def JLIR_add_ptr   : JLIR_ArithmeticOp<"add_ptr">;
def JLIR_sub_ptr   : JLIR_ArithmeticOp<"sub_ptr">; 
def JLIR_neg_float : JLIR_UnaryArithmeticOp<"neg_float">;
def JLIR_add_float : JLIR_ArithmeticOp<"add_float">; // ??? Commutative
def JLIR_sub_float : JLIR_ArithmeticOp<"sub_float">; 
def JLIR_mul_float : JLIR_ArithmeticOp<"mul_float">; // ??? Commutative
def JLIR_div_float : JLIR_ArithmeticOp<"div_float">;
def JLIR_rem_float : JLIR_ArithmeticOp<"rem_float">;
// 3-ary arithmetic ops
def JLIR_fma_float    : JLIR_IntrinsicOp<"fma_float", [NoSideEffect]>; // FIXME: SameOperandsAndResultType
def JLIR_muladd_float : JLIR_IntrinsicOp<"muladd_float", [NoSideEffect]>; // FIXME: SameOperandsAndResultType
//    /*  fast arithmetic */ \
//    ALIAS(neg_float_fast, neg_float) \
//    ALIAS(add_float_fast, add_float) \
//    ALIAS(sub_float_fast, sub_float) \
//    ALIAS(mul_float_fast, mul_float) \
//    ALIAS(div_float_fast, div_float) \
//    ALIAS(rem_float_fast, rem_float) \
//  same-type comparisons
def JLIR_eq_int : JLIR_IntrinsicOp<"eq_int">; //  2
def JLIR_ne_int : JLIR_IntrinsicOp<"ne_int">; //  2
def JLIR_slt_int : JLIR_IntrinsicOp<"slt_int">; //  2
def JLIR_ult_int : JLIR_IntrinsicOp<"ult_int">; //  2
def JLIR_sle_int : JLIR_IntrinsicOp<"sle_int">; //  2
def JLIR_ule_int : JLIR_IntrinsicOp<"ule_int">; //  2
def JLIR_eq_float : JLIR_IntrinsicOp<"eq_float">; //  2
def JLIR_ne_float : JLIR_IntrinsicOp<"ne_float">; //  2
def JLIR_lt_float : JLIR_IntrinsicOp<"lt_float">; //  2
def JLIR_le_float : JLIR_IntrinsicOp<"le_float">; //  2
//    ALIAS(eq_float_fast, eq_float) \
//    ALIAS(ne_float_fast, ne_float) \
//    ALIAS(lt_float_fast, lt_float) \
//    ALIAS(le_float_fast, le_float) \
def JLIR_fpiseq : JLIR_IntrinsicOp<"fpiseq">; //  2
def JLIR_fpislt : JLIR_IntrinsicOp<"fpislt">; //  2
// /*  bitwise operators */ \
def JLIR_and_int : JLIR_IntrinsicOp<"and_int">; //  2
def JLIR_or_int : JLIR_IntrinsicOp<"or_int">; //  2
def JLIR_xor_int : JLIR_IntrinsicOp<"xor_int">; //  2
def JLIR_not_int : JLIR_IntrinsicOp<"not_int">; //  1
def JLIR_shl_int : JLIR_IntrinsicOp<"shl_int">; //  2
def JLIR_lshr_int : JLIR_IntrinsicOp<"lshr_int">; //  2
def JLIR_ashr_int : JLIR_IntrinsicOp<"ashr_int">; //  2
def JLIR_bswap_int : JLIR_IntrinsicOp<"bswap_int">; //  1
def JLIR_ctpop_int : JLIR_IntrinsicOp<"ctpop_int">; //  1
def JLIR_ctlz_int : JLIR_IntrinsicOp<"ctlz_int">; //  1
def JLIR_cttz_int : JLIR_IntrinsicOp<"cttz_int">; //  1
// /*  conversion */ \
def JLIR_sext_int : JLIR_IntrinsicOp<"sext_int">; //  2
def JLIR_zext_int : JLIR_IntrinsicOp<"zext_int">; //  2
def JLIR_trunc_int : JLIR_IntrinsicOp<"trunc_int">; //  2
def JLIR_fptoui : JLIR_IntrinsicOp<"fptoui">; //  2
def JLIR_fptosi : JLIR_IntrinsicOp<"fptosi">; //  2
def JLIR_uitofp : JLIR_IntrinsicOp<"uitofp">; //  2
def JLIR_sitofp : JLIR_IntrinsicOp<"sitofp">; //  2
def JLIR_fptrunc : JLIR_IntrinsicOp<"fptrunc">; //  2
def JLIR_fpext : JLIR_IntrinsicOp<"fpext">; //  2
//  checked arithmetic -- have side-effect e.g. throw an error
def JLIR_checked_sadd_int : JLIR_IntrinsicOp<"checked_sadd_int">; //  2
def JLIR_checked_uadd_int : JLIR_IntrinsicOp<"checked_uadd_int">; //  2
def JLIR_checked_ssub_int : JLIR_IntrinsicOp<"checked_ssub_int">; //  2
def JLIR_checked_usub_int : JLIR_IntrinsicOp<"checked_usub_int">; //  2
def JLIR_checked_smul_int : JLIR_IntrinsicOp<"checked_smul_int">; //  2
def JLIR_checked_umul_int : JLIR_IntrinsicOp<"checked_umul_int">; //  2
def JLIR_checked_sdiv_int : JLIR_IntrinsicOp<"checked_sdiv_int">; //  2
def JLIR_checked_udiv_int : JLIR_IntrinsicOp<"checked_udiv_int">; //  2
def JLIR_checked_srem_int : JLIR_IntrinsicOp<"checked_srem_int">; //  2
def JLIR_checked_urem_int : JLIR_IntrinsicOp<"checked_urem_int">; //  2
// functions
def JLIR_abs_float : JLIR_IntrinsicOp<"abs_float">; //  1
def JLIR_copysign_float : JLIR_IntrinsicOp<"copysign_float">; //  2
def JLIR_flipsign_int : JLIR_IntrinsicOp<"flipsign_int">; //  2
def JLIR_ceil_llvm : JLIR_IntrinsicOp<"ceil_llvm">; //  1
def JLIR_floor_llvm : JLIR_IntrinsicOp<"floor_llvm">; //  1
def JLIR_trunc_llvm : JLIR_IntrinsicOp<"trunc_llvm">; //  1
def JLIR_rint_llvm : JLIR_IntrinsicOp<"rint_llvm">; //  1
def JLIR_sqrt_llvm : JLIR_IntrinsicOp<"sqrt_llvm">; //  1
def JLIR_sqrt_llvm_fast : JLIR_IntrinsicOp<"sqrt_llvm_fast">; //  1
// /*  pointer access */ \
def JLIR_pointerref : JLIR_IntrinsicOp<"pointerref">; //  3
def JLIR_pointerset : JLIR_IntrinsicOp<"pointerset">; //  4
// /* c interface */ \
def JLIR_cglobal : JLIR_IntrinsicOp<"cglobal">; //  2
// ALIAS(llvmcall, llvmcall) \
// /* object access */ \
def JLIR_arraylen : JLIR_IntrinsicOp<"arraylen">; //  1
// /*  hidden intrinsics */ \
// ADD_HIDDEN(cglobal_auto, 1)

#endif // JULIA_MLIR_JLIR_TD
