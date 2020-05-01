# RUN: julia --startup-file=no %s 2>&1 | FileCheck %s

import Brutus

emit_lowered(f, tt...) =
    Brutus.emit(typeof(f), tt,
                emit_llvm=false, # TODO: change to true when ready
                dump_options=[Brutus.DumpLoweredToStd,
                              Brutus.DumpLoweredToLLVM])

emit_lowered(identity, Bool)
# CHECK: func @identity(%arg0: !jlir<"typeof(Base.identity)">, %arg1: i1) -> i1
# CHECK:   return %arg1 : i1
# CHECK: llvm.func @identity(%arg0: !llvm<"%jl_value_t*">, %arg1: !llvm.i1) -> !llvm.i1
# CHECK:   llvm.return %arg1 : !llvm.i1
emit_lowered(identity, Nothing)
# CHECK: func @identity(%arg0: !jlir<"typeof(Base.identity)">, %arg1: !jlir.Nothing) -> !jlir.Nothing
# CHECK:   "jlir.return"(%arg1) : (!jlir.Nothing) -> ()
# CHECK: llvm.func @identity(%arg0: !llvm<"%jl_value_t*">, %arg1: !llvm<"%jl_value_t*">) -> !llvm<"%jl_value_t*">
# CHECK:   llvm.return %arg1 : !llvm<"%jl_value_t*">
emit_lowered(identity, Any)
# CHECK: func @identity(%arg0: !jlir<"typeof(Base.identity)">, %arg1: !jlir.Any) -> !jlir.Any
# CHECK:   "jlir.return"(%arg1) : (!jlir.Any) -> ()
# CHECK: llvm.func @identity(%arg0: !llvm<"%jl_value_t*">, %arg1: !llvm<"%jl_value_t*">) -> !llvm<"%jl_value_t*">
# CHECK:   llvm.return %arg1 : !llvm<"%jl_value_t*">

add(x, y) = x + y
emit_lowered(add, Int64, Int64)
# CHECK: func @add(%arg0: !jlir<"typeof(Main.add)">, %arg1: i64, %arg2: i64) -> i64
# CHECK:   %0 = addi %arg1, %arg2 : i64
# CHECK:   return %0 : i64
# CHECK: llvm.func @add(%arg0: !llvm<"%jl_value_t*">, %arg1: !llvm.i64, %arg2: !llvm.i64) -> !llvm.i64
# CHECK:   %0 = llvm.add %arg1, %arg2 : !llvm.i64
# CHECK:   llvm.return %0 : !llvm.i64
emit_lowered(add, Float64, Float64)
# CHECK: func @add(%arg0: !jlir<"typeof(Main.add)">, %arg1: f64, %arg2: f64) -> f64
# CHECK:   %0 = addf %arg1, %arg2 : f64
# CHECK:   return %0 : f64
# CHECK: llvm.func @add(%arg0: !llvm<"%jl_value_t*">, %arg1: !llvm.double, %arg2: !llvm.double) -> !llvm.double
# CHECK:   %0 = llvm.fadd %arg1, %arg2 : !llvm.double
# CHECK:   llvm.return %0 : !llvm.double

sle_int(x, y) = Base.sle_int(x, y)
emit_lowered(sle_int, Int64, Int64)
# CHECK: func @sle_int(%arg0: !jlir<"typeof(Main.sle_int)">, %arg1: i64, %arg2: i64) -> i1
# CHECK:   %0 = cmpi "sle", %arg1, %arg2 : i64
# CHECK:   return %0 : i1
# CHECK: llvm.func @sle_int(%arg0: !llvm<"%jl_value_t*">, %arg1: !llvm.i64, %arg2: !llvm.i64) -> !llvm.i1
# CHECK:   %0 = llvm.icmp "sle" %arg1, %arg2 : !llvm.i64
# CHECK:   llvm.return %0 : !llvm.i1

ne(x, y) = x != y
emit_lowered(ne, Float64, Float64)
# CHECK: func @ne(%arg0: !jlir<"typeof(Main.ne)">, %arg1: f64, %arg2: f64) -> i1
# CHECK:   %0 = cmpf "une", %arg1, %arg2 : f64
# CHECK:   return %0 : i1
# CHECK: llvm.func @ne(%arg0: !llvm<"%jl_value_t*">, %arg1: !llvm.double, %arg2: !llvm.double) -> !llvm.i1
# CHECK:   %0 = llvm.fcmp "une" %arg1, %arg2 : !llvm.double
# CHECK:   llvm.return %0 : !llvm.i1

symbol() = :testing
emit_lowered(symbol)
# CHECK: func @symbol(%arg0: !jlir<"typeof(Main.symbol)">) -> !jlir.Symbol
# CHECK:   %0 = "jlir.constant"() {value = #jlir<":testing">} : () -> !jlir.Symbol
# CHECK:   "jlir.return"(%0) : (!jlir.Symbol) -> ()
# CHECK: llvm.func @symbol(%arg0: !llvm<"%jl_value_t*">) -> !llvm<"%jl_value_t*"> {
# CHECK:   %0 = llvm.mlir.constant({{[0-9]+}} : i64) : !llvm.i64
# CHECK:   %1 = llvm.inttoptr %0 : !llvm.i64 to !llvm<"%jl_value_t*">
# CHECK:   llvm.return %1 : !llvm<"%jl_value_t*">

select(c) = 1 + (c ? 2 : 3)
emit_lowered(select, Bool)
# CHECK: func @select(%arg0: !jlir<"typeof(Main.select)">, %arg1: i1) -> i64
# CHECK:   %c2_i64 = constant 2 : i64
# CHECK:   %c3_i64 = constant 3 : i64
# CHECK:   %c1_i64 = constant 1 : i64
# CHECK:   %0 = select %arg1, %c2_i64, %c3_i64 : i64
# CHECK:   %1 = addi %0, %c1_i64 : i64
# CHECK:   return %1 : i64
# CHECK: llvm.func @select(%arg0: !llvm<"%jl_value_t*">, %arg1: !llvm.i1) -> !llvm.i64
# CHECK:   %0 = llvm.mlir.constant(2 : i64) : !llvm.i64
# CHECK:   %1 = llvm.mlir.constant(3 : i64) : !llvm.i64
# CHECK:   %2 = llvm.mlir.constant(1 : i64) : !llvm.i64
# CHECK:   %3 = llvm.select %arg1, %0, %1 : !llvm.i1, !llvm.i64
# CHECK:   %4 = llvm.add %3, %2 : !llvm.i64
# CHECK:   llvm.return %4 : !llvm.i64

function loop(N)
    acc = 0
    for i in 1:N
        acc += i
    end
    return acc
end
emit_lowered(loop, Int64)
# CHECK: func @loop(%arg0: !jlir<"typeof(Main.loop)">, %arg1: i64) -> i64
# CHECK:   %c1_i64 = constant 1 : i64
# CHECK:   %c0_i64 = constant 0 : i64
# CHECK:   %false = constant 0 : i1
# CHECK:   %true = constant 1 : i1
# CHECK:   %0 = cmpi "sle", %c1_i64, %arg1 : i64
# CHECK:   %1 = select %0, %arg1, %c0_i64 : i64
# CHECK:   %2 = "jlir.convertstd"(%1) : (i64) -> !jlir.Int64
# CHECK:   %3 = cmpi "slt", %1, %c1_i64 : i64
# CHECK:   cond_br %3, ^bb1, ^bb2(%false, %c1_i64, %c1_i64 : i1, i64, i64)
# CHECK: ^bb1:
# CHECK:   %4 = "jlir.undef"() : () -> !jlir.Int64
# CHECK:   %5 = "jlir.undef"() : () -> !jlir.Int64
# CHECK:   %6 = "jlir.convertstd"(%4) : (!jlir.Int64) -> i64
# CHECK:   %7 = "jlir.convertstd"(%5) : (!jlir.Int64) -> i64
# CHECK:   br ^bb2(%true, %6, %7 : i1, i64, i64)
# CHECK: ^bb2(%8: i1, %9: i64, %10: i64):
# CHECK:   %11 = xor %8, %true : i1
# CHECK:   cond_br %11, ^bb3(%c0_i64, %9, %10 : i64, i64, i64), ^bb7(%c0_i64 : i64)
# CHECK: ^bb3(%12: i64, %13: i64, %14: i64):
# CHECK:   %15 = "jlir.convertstd"(%14) : (i64) -> !jlir.Int64
# CHECK:   %16 = addi %12, %13 : i64
# CHECK:   %17 = "jlir.==="(%15, %2) : (!jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK:   %18 = "jlir.convertstd"(%17) : (!jlir.Bool) -> i1
# CHECK:   cond_br %18, ^bb4, ^bb5
# CHECK: ^bb4:
# CHECK:   %19 = "jlir.undef"() : () -> !jlir.Int64
# CHECK:   %20 = "jlir.undef"() : () -> !jlir.Int64
# CHECK:   %21 = "jlir.convertstd"(%19) : (!jlir.Int64) -> i64
# CHECK:   %22 = "jlir.convertstd"(%20) : (!jlir.Int64) -> i64
# CHECK:   br ^bb6(%21, %22, %true : i64, i64, i1)
# CHECK: ^bb5:
# CHECK:   %23 = addi %14, %c1_i64 : i64
# CHECK:   br ^bb6(%23, %23, %false : i64, i64, i1)
# CHECK: ^bb6(%24: i64, %25: i64, %26: i1):
# CHECK:   %27 = xor %26, %true : i1
# CHECK:   cond_br %27, ^bb3(%16, %24, %25 : i64, i64, i64), ^bb7(%16 : i64)
# CHECK: ^bb7(%28: i64):
# CHECK:   return %28 : i64
#
# CHECK: llvm.func @loop(%arg0: !llvm<"%jl_value_t*">, %arg1: !llvm.i64) -> !llvm.i64
# CHECK:   %0 = llvm.mlir.constant(1 : i64) : !llvm.i64
# CHECK:   %1 = llvm.mlir.constant(0 : i64) : !llvm.i64
# CHECK:   %2 = llvm.mlir.constant(0 : i1) : !llvm.i1
# CHECK:   %3 = llvm.mlir.constant(1 : i1) : !llvm.i1
# CHECK:   %4 = llvm.icmp "sle" %0, %arg1 : !llvm.i64
# CHECK:   %5 = llvm.select %4, %arg1, %1 : !llvm.i1, !llvm.i64
# CHECK:   %6 = llvm.icmp "slt" %5, %0 : !llvm.i64
# CHECK:   llvm.cond_br %6, ^bb1, ^bb2(%2, %0, %0 : !llvm.i1, !llvm.i64, !llvm.i64)
# CHECK: ^bb1:
# CHECK:   %7 = llvm.mlir.undef : !llvm.i64
# CHECK:   %8 = llvm.mlir.undef : !llvm.i64
# CHECK:   llvm.br ^bb2(%3, %7, %8 : !llvm.i1, !llvm.i64, !llvm.i64)
# CHECK: ^bb2(%9: !llvm.i1, %10: !llvm.i64, %11: !llvm.i64):
# CHECK:   %12 = llvm.xor %9, %3 : !llvm.i1
# CHECK:   llvm.cond_br %12, ^bb3(%1, %10, %11 : !llvm.i64, !llvm.i64, !llvm.i64), ^bb7(%1 : !llvm.i64)
# CHECK: ^bb3(%13: !llvm.i64, %14: !llvm.i64, %15: !llvm.i64):
# CHECK:   %16 = llvm.add %13, %14 : !llvm.i64
# CHECK:   %17 = llvm.icmp "eq" %15, %5 : !llvm.i64
# CHECK:   llvm.cond_br %17, ^bb4, ^bb5
# CHECK: ^bb4:
# CHECK:   %18 = llvm.mlir.undef : !llvm.i64
# CHECK:   %19 = llvm.mlir.undef : !llvm.i64
# CHECK:   llvm.br ^bb6(%18, %19, %3 : !llvm.i64, !llvm.i64, !llvm.i1)
# CHECK: ^bb5:
# CHECK:   %20 = llvm.add %15, %0 : !llvm.i64
# CHECK:   llvm.br ^bb6(%20, %20, %2 : !llvm.i64, !llvm.i64, !llvm.i1)
# CHECK: ^bb6(%21: !llvm.i64, %22: !llvm.i64, %23: !llvm.i1):
# CHECK:   %24 = llvm.xor %23, %3 : !llvm.i1
# CHECK:   llvm.cond_br %24, ^bb3(%16, %21, %22 : !llvm.i64, !llvm.i64, !llvm.i64), ^bb7(%16 : !llvm.i64)
# CHECK: ^bb7(%25: !llvm.i64):
# CHECK:   llvm.return %25 : !llvm.i64
