//===----------------------------------------------------------------------===//
//
// Defines the operations of the Julia dialect.
//
//===----------------------------------------------------------------------===//

#ifndef JULIA_MLIR_JLIR_TD
#define JULIA_MLIR_JLIR_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffects.td"

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
class JLIR_IntrinsicBuiltinOp<string mnemonic, list<OpTrait> traits = []> :
    JLIR_Op<mnemonic, traits> {

    let arguments = (ins Variadic<JLIR_JuliaType>:$arguments);
    let results = (outs JLIR_JuliaType);

    let builders = [
        OpBuilder<"Builder *builder, OperationState &state,"
                  "jl_datatype_t *type, ArrayRef<Value> arguments">
    ];
}

class JLIR_ArithmeticOp<string mnemonic, list<OpTrait> traits = []>:
    JLIR_IntrinsicBuiltinOp<mnemonic, !listconcat([NoSideEffect, SameOperandsAndResultType], traits)> {

    let arguments = (ins JLIR_JuliaType:$rhs, JLIR_JuliaType:$lhs);
    let results = (outs JLIR_JuliaType);
}

class JLIR_UnaryArithmeticOp<string mnemonic, list<OpTrait> traits = []>:
    JLIR_IntrinsicBuiltinOp<mnemonic, !listconcat([NoSideEffect, SameOperandsAndResultType], traits)> {

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

    let hasCanonicalizer = 1;
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
    let successors = (successor AnySuccessor:$dest);

    let builders = [
        OpBuilder<"Builder *builder, OperationState &state, Block *dest,"
                  "ValueRange operands = {}",
                  [{ build(builder, state, operands, dest); }]>
    ];
}

def GotoIfNotOp : JLIR_Op<"gotoifnot", [AttrSizedOperandSegments, Terminator]> {
    let summary = "goto if not operation";
    let description = [{
        TODO
    }];

    let arguments = (ins JLIR_JuliaType:$condition,
                     Variadic<JLIR_JuliaType>:$branchOperands,
                     Variadic<JLIR_JuliaType>:$fallthroughOperands);
    let successors = (successor AnySuccessor:$falseDest, AnySuccessor:$trueDest);

    let builders = [
        OpBuilder<"Builder *builder, OperationState &state, Value condition,"
                  "Block *branchDest, ValueRange branchOperands,"
                  "Block *fallthroughDest, ValueRange fallthroughOperands",
                  [{
                      build(builder, state, condition,
                            branchOperands, fallthroughOperands,
                            branchDest, fallthroughDest);
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
def JLIR_Intrinsic_bitcast : JLIR_IntrinsicBuiltinOp<"bitcast", [NoSideEffect]>;

/*  arithmetic */
def JLIR_Intrinsic_neg_int   : JLIR_UnaryArithmeticOp<"neg_int">;
def JLIR_Intrinsic_add_int   : JLIR_ArithmeticOp<"add_int", [Commutative]>;
def JLIR_Intrinsic_sub_int   : JLIR_ArithmeticOp<"sub_int">;
def JLIR_Intrinsic_mul_int   : JLIR_ArithmeticOp<"mul_int", [Commutative]>;
def JLIR_Intrinsic_sdiv_int  : JLIR_ArithmeticOp<"sdiv_int">;
def JLIR_Intrinsic_udiv_int  : JLIR_ArithmeticOp<"udiv_int">;
def JLIR_Intrinsic_srem_int  : JLIR_ArithmeticOp<"srem_int">;
def JLIR_Intrinsic_urem_int  : JLIR_ArithmeticOp<"urem_int">;
def JLIR_Intrinsic_add_ptr   : JLIR_ArithmeticOp<"add_ptr">;
def JLIR_Intrinsic_sub_ptr   : JLIR_ArithmeticOp<"sub_ptr">;
def JLIR_Intrinsic_neg_float : JLIR_UnaryArithmeticOp<"neg_float">;
def JLIR_Intrinsic_add_float : JLIR_ArithmeticOp<"add_float">;
def JLIR_Intrinsic_sub_float : JLIR_ArithmeticOp<"sub_float">;
def JLIR_Intrinsic_mul_float : JLIR_ArithmeticOp<"mul_float">;
def JLIR_Intrinsic_div_float : JLIR_ArithmeticOp<"div_float">;
def JLIR_Intrinsic_rem_float : JLIR_ArithmeticOp<"rem_float">;
// 3-ary arithmetic ops
def JLIR_Intrinsic_fma_float    : JLIR_IntrinsicBuiltinOp<"fma_float", [NoSideEffect]>; // FIXME: SameOperandsAndResultType
def JLIR_Intrinsic_muladd_float : JLIR_IntrinsicBuiltinOp<"muladd_float", [NoSideEffect]>; // FIXME: SameOperandsAndResultType
//    /*  fast arithmetic */ \
//    ALIAS(neg_float_fast, neg_float) \
//    ALIAS(add_float_fast, add_float) \
//    ALIAS(sub_float_fast, sub_float) \
//    ALIAS(mul_float_fast, mul_float) \
//    ALIAS(div_float_fast, div_float) \
//    ALIAS(rem_float_fast, rem_float) \
//  same-type comparisons
def JLIR_Intrinsic_eq_int : JLIR_IntrinsicBuiltinOp<"eq_int">; //  2
def JLIR_Intrinsic_ne_int : JLIR_IntrinsicBuiltinOp<"ne_int">; //  2
def JLIR_Intrinsic_slt_int : JLIR_IntrinsicBuiltinOp<"slt_int">; //  2
def JLIR_Intrinsic_ult_int : JLIR_IntrinsicBuiltinOp<"ult_int">; //  2
def JLIR_Intrinsic_sle_int : JLIR_IntrinsicBuiltinOp<"sle_int">; //  2
def JLIR_Intrinsic_ule_int : JLIR_IntrinsicBuiltinOp<"ule_int">; //  2
def JLIR_Intrinsic_eq_float : JLIR_IntrinsicBuiltinOp<"eq_float">; //  2
def JLIR_Intrinsic_ne_float : JLIR_IntrinsicBuiltinOp<"ne_float">; //  2
def JLIR_Intrinsic_lt_float : JLIR_IntrinsicBuiltinOp<"lt_float">; //  2
def JLIR_Intrinsic_le_float : JLIR_IntrinsicBuiltinOp<"le_float">; //  2
//    ALIAS(eq_float_fast, eq_float) \
//    ALIAS(ne_float_fast, ne_float) \
//    ALIAS(lt_float_fast, lt_float) \
//    ALIAS(le_float_fast, le_float) \
def JLIR_Intrinsic_fpiseq : JLIR_IntrinsicBuiltinOp<"fpiseq">; //  2
def JLIR_Intrinsic_fpislt : JLIR_IntrinsicBuiltinOp<"fpislt">; //  2
// /*  bitwise operators */ \
def JLIR_Intrinsic_and_int : JLIR_IntrinsicBuiltinOp<"and_int">; //  2
def JLIR_Intrinsic_or_int : JLIR_IntrinsicBuiltinOp<"or_int">; //  2
def JLIR_Intrinsic_xor_int : JLIR_IntrinsicBuiltinOp<"xor_int">; //  2
def JLIR_Intrinsic_not_int : JLIR_IntrinsicBuiltinOp<"not_int">; //  1
def JLIR_Intrinsic_shl_int : JLIR_IntrinsicBuiltinOp<"shl_int">; //  2
def JLIR_Intrinsic_lshr_int : JLIR_IntrinsicBuiltinOp<"lshr_int">; //  2
def JLIR_Intrinsic_ashr_int : JLIR_IntrinsicBuiltinOp<"ashr_int">; //  2
def JLIR_Intrinsic_bswap_int : JLIR_IntrinsicBuiltinOp<"bswap_int">; //  1
def JLIR_Intrinsic_ctpop_int : JLIR_IntrinsicBuiltinOp<"ctpop_int">; //  1
def JLIR_Intrinsic_ctlz_int : JLIR_IntrinsicBuiltinOp<"ctlz_int">; //  1
def JLIR_Intrinsic_cttz_int : JLIR_IntrinsicBuiltinOp<"cttz_int">; //  1
// /*  conversion */ \
def JLIR_Intrinsic_sext_int : JLIR_IntrinsicBuiltinOp<"sext_int">; //  2
def JLIR_Intrinsic_zext_int : JLIR_IntrinsicBuiltinOp<"zext_int">; //  2
def JLIR_Intrinsic_trunc_int : JLIR_IntrinsicBuiltinOp<"trunc_int">; //  2
def JLIR_Intrinsic_fptoui : JLIR_IntrinsicBuiltinOp<"fptoui">; //  2
def JLIR_Intrinsic_fptosi : JLIR_IntrinsicBuiltinOp<"fptosi">; //  2
def JLIR_Intrinsic_uitofp : JLIR_IntrinsicBuiltinOp<"uitofp">; //  2
def JLIR_Intrinsic_sitofp : JLIR_IntrinsicBuiltinOp<"sitofp">; //  2
def JLIR_Intrinsic_fptrunc : JLIR_IntrinsicBuiltinOp<"fptrunc">; //  2
def JLIR_Intrinsic_fpext : JLIR_IntrinsicBuiltinOp<"fpext">; //  2
//  checked arithmetic -- have side-effect e.g. throw an error
def JLIR_Intrinsic_checked_sadd_int : JLIR_IntrinsicBuiltinOp<"checked_sadd_int">; //  2
def JLIR_Intrinsic_checked_uadd_int : JLIR_IntrinsicBuiltinOp<"checked_uadd_int">; //  2
def JLIR_Intrinsic_checked_ssub_int : JLIR_IntrinsicBuiltinOp<"checked_ssub_int">; //  2
def JLIR_Intrinsic_checked_usub_int : JLIR_IntrinsicBuiltinOp<"checked_usub_int">; //  2
def JLIR_Intrinsic_checked_smul_int : JLIR_IntrinsicBuiltinOp<"checked_smul_int">; //  2
def JLIR_Intrinsic_checked_umul_int : JLIR_IntrinsicBuiltinOp<"checked_umul_int">; //  2
def JLIR_Intrinsic_checked_sdiv_int : JLIR_IntrinsicBuiltinOp<"checked_sdiv_int">; //  2
def JLIR_Intrinsic_checked_udiv_int : JLIR_IntrinsicBuiltinOp<"checked_udiv_int">; //  2
def JLIR_Intrinsic_checked_srem_int : JLIR_IntrinsicBuiltinOp<"checked_srem_int">; //  2
def JLIR_Intrinsic_checked_urem_int : JLIR_IntrinsicBuiltinOp<"checked_urem_int">; //  2
// functions
def JLIR_Intrinsic_abs_float : JLIR_IntrinsicBuiltinOp<"abs_float">; //  1
def JLIR_Intrinsic_copysign_float : JLIR_IntrinsicBuiltinOp<"copysign_float">; //  2
def JLIR_Intrinsic_flipsign_int : JLIR_IntrinsicBuiltinOp<"flipsign_int">; //  2
def JLIR_Intrinsic_ceil_llvm : JLIR_IntrinsicBuiltinOp<"ceil_llvm">; //  1
def JLIR_Intrinsic_floor_llvm : JLIR_IntrinsicBuiltinOp<"floor_llvm">; //  1
def JLIR_Intrinsic_trunc_llvm : JLIR_IntrinsicBuiltinOp<"trunc_llvm">; //  1
def JLIR_Intrinsic_rint_llvm : JLIR_IntrinsicBuiltinOp<"rint_llvm">; //  1
def JLIR_Intrinsic_sqrt_llvm : JLIR_IntrinsicBuiltinOp<"sqrt_llvm">; //  1
def JLIR_Intrinsic_sqrt_llvm_fast : JLIR_IntrinsicBuiltinOp<"sqrt_llvm_fast">; //  1
// /*  pointer access */ \
def JLIR_Intrinsic_pointerref : JLIR_IntrinsicBuiltinOp<"pointerref">; //  3
def JLIR_Intrinsic_pointerset : JLIR_IntrinsicBuiltinOp<"pointerset">; //  4
// /* c interface */ \
def JLIR_Intrinsic_cglobal : JLIR_IntrinsicBuiltinOp<"cglobal">; //  2
// ALIAS(llvmcall, llvmcall) \
// /* object access */ \
def JLIR_Intrinsic_arraylen : JLIR_IntrinsicBuiltinOp<"arraylen">; //  1
// /*  hidden intrinsics */ \
// ADD_HIDDEN(cglobal_auto, 1)

/* builtins */
def JLIR_Builtin_throw : JLIR_IntrinsicBuiltinOp<"throw">;
def JLIR_Builtin_is : JLIR_IntrinsicBuiltinOp<"is">;
def JLIR_Builtin_typeof : JLIR_IntrinsicBuiltinOp<"typeof">;
def JLIR_Builtin_sizeof : JLIR_IntrinsicBuiltinOp<"sizeof">;
def JLIR_Builtin_issubtype : JLIR_IntrinsicBuiltinOp<"issubtype">;
def JLIR_Builtin_isa : JLIR_IntrinsicBuiltinOp<"isa">;
def JLIR_Builtin__apply : JLIR_IntrinsicBuiltinOp<"_apply">;
def JLIR_Builtin__apply_pure : JLIR_IntrinsicBuiltinOp<"_apply_pure">;
def JLIR_Builtin__apply_latest : JLIR_IntrinsicBuiltinOp<"_apply_latest">;
def JLIR_Builtin__apply_iterate : JLIR_IntrinsicBuiltinOp<"_apply_iterate">;
def JLIR_Builtin_isdefined : JLIR_IntrinsicBuiltinOp<"isdefined">;
def JLIR_Builtin_nfields : JLIR_IntrinsicBuiltinOp<"nfields">;
def JLIR_Builtin_tuple : JLIR_IntrinsicBuiltinOp<"tuple">;
def JLIR_Builtin_svec : JLIR_IntrinsicBuiltinOp<"svec">;
def JLIR_Builtin_getfield : JLIR_IntrinsicBuiltinOp<"getfield">;
def JLIR_Builtin_setfield : JLIR_IntrinsicBuiltinOp<"setfield">;
def JLIR_Builtin_fieldtype : JLIR_IntrinsicBuiltinOp<"fieldtype">;
def JLIR_Builtin_arrayref : JLIR_IntrinsicBuiltinOp<"arrayref">;
def JLIR_Builtin_const_arrayref : JLIR_IntrinsicBuiltinOp<"const_arrayref">;
def JLIR_Builtin_arrayset : JLIR_IntrinsicBuiltinOp<"arrayset">;
def JLIR_Builtin_arraysize : JLIR_IntrinsicBuiltinOp<"arraysize">;
def JLIR_Builtin_apply_type : JLIR_IntrinsicBuiltinOp<"apply_type">;
def JLIR_Builtin_applicable : JLIR_IntrinsicBuiltinOp<"applicable">;
// def JLIR_Builtin_invoke : JLIR_IntrinsicBuiltinOp<"invoke">;
def JLIR_Builtin__expr : JLIR_IntrinsicBuiltinOp<"_expr">;
def JLIR_Builtin_typeassert : JLIR_IntrinsicBuiltinOp<"typeassert">;
def JLIR_Builtin_ifelse : JLIR_IntrinsicBuiltinOp<"ifelse">;
def JLIR_Builtin__typevar : JLIR_IntrinsicBuiltinOp<"_typevar">;
// invoke_kwsorter?

#endif // JULIA_MLIR_JLIR_TD
