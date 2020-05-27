# RUN: julia --startup-file=no %s 2>&1 | FileCheck --dump-input=fail %s
import Brutus

emit_lowered(f, tt...) =
    Brutus.emit(typeof(f), tt,
                emit_fptr=false, # TODO: change to true when ready
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

function gauss(N)
    acc = 0
    for i in 1:N
        acc += i
    end
    return acc
end
emit_lowered(gauss, Int64)
# CHECK: func @gauss(%arg0: !jlir<"typeof(Main.gauss)">, %arg1: i64) -> i64
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
# CHECK: llvm.func @gauss(%arg0: !llvm<"%jl_value_t*">, %arg1: !llvm.i64) -> !llvm.i64
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
# CHECK:   llvm.br ^bb2(%3, %7, %7 : !llvm.i1, !llvm.i64, !llvm.i64)
# CHECK: ^bb2(%8: !llvm.i1, %9: !llvm.i64, %10: !llvm.i64):
# CHECK:   %11 = llvm.xor %8, %3 : !llvm.i1
# CHECK:   llvm.cond_br %11, ^bb3(%1, %9, %10 : !llvm.i64, !llvm.i64, !llvm.i64), ^bb7(%1 : !llvm.i64)
# CHECK: ^bb3(%12: !llvm.i64, %13: !llvm.i64, %14: !llvm.i64):
# CHECK:   %15 = llvm.add %12, %13 : !llvm.i64
# CHECK:   %16 = llvm.icmp "eq" %14, %5 : !llvm.i64
# CHECK:   llvm.cond_br %16, ^bb4, ^bb5
# CHECK: ^bb4:
# CHECK:   %17 = llvm.mlir.undef : !llvm.i64
# CHECK:   llvm.br ^bb6(%17, %17, %3 : !llvm.i64, !llvm.i64, !llvm.i1)
# CHECK: ^bb5:
# CHECK:   %18 = llvm.add %14, %0 : !llvm.i64
# CHECK:   llvm.br ^bb6(%18, %18, %2 : !llvm.i64, !llvm.i64, !llvm.i1)
# CHECK: ^bb6(%19: !llvm.i64, %20: !llvm.i64, %21: !llvm.i1):
# CHECK:   %22 = llvm.xor %21, %3 : !llvm.i1
# CHECK:   llvm.cond_br %22, ^bb3(%15, %19, %20 : !llvm.i64, !llvm.i64, !llvm.i64), ^bb7(%15 : !llvm.i64)
# CHECK: ^bb7(%23: !llvm.i64):
# CHECK:   llvm.return %23 : !llvm.i64

index(A, i) = A[i]
emit_lowered(index, Array{Int64, 1}, Int64)
# CHECK: func @index(%arg0: !jlir<"typeof(Main.index)">, %arg1: !jlir<"Array{Int64, 1}">, %arg2: i64) -> i64
# CHECK:   %c1 = constant 1 : index
# CHECK:   %0 = "jlir.convertstd"(%arg2) : (i64) -> index
# CHECK:   %1 = subi %0, %c1 : index
# CHECK:   %2 = "jlir.arraytomemref"(%arg1) : (!jlir<"Array{Int64, 1}">) -> memref<?xi64>
# CHECK:   %3 = load %2[%1] : memref<?xi64>
# CHECK:   return %3 : i64
#
# CHECK: llvm.func @index(%arg0: !llvm<"%jl_value_t*">, %arg1: !llvm<"%jl_value_t*">, %arg2: !llvm.i64) -> !llvm.i64
# CHECK:   %0 = llvm.mlir.constant(1 : index) : !llvm.i64
# CHECK:   %1 = llvm.sub %arg2, %0 : !llvm.i64
# CHECK:   %2 = llvm.bitcast %arg1 : !llvm<"%jl_value_t*"> to !llvm<"%jl_array_t*">
# CHECK:   %3 = llvm.mlir.constant(0 : i64) : !llvm.i64
# CHECK:   %4 = llvm.mlir.constant(0 : i32) : !llvm.i32
# CHECK:   %5 = llvm.getelementptr %2[%3, %4] : (!llvm<"%jl_array_t*">, !llvm.i64, !llvm.i32) -> !llvm<"i64*">
# CHECK:   %6 = llvm.load %5 : !llvm<"i64*">
# CHECK:   %7 = llvm.bitcast %6 : !llvm<"i8*"> to !llvm<"i64*">
# CHECK:   %8 = llvm.mlir.constant(5 : i32) : !llvm.i32
# CHECK:   %9 = llvm.getelementptr %2[%3, %8] : (!llvm<"%jl_array_t*">, !llvm.i64, !llvm.i32) -> !llvm<"i64*">
# CHECK:   %10 = llvm.mlir.undef : !llvm<"{ i64*, i64*, i64, [1 x i64], [1 x i64] }">
# CHECK:   %11 = llvm.insertvalue %7, %10[0] : !llvm<"{ i64*, i64*, i64, [1 x i64], [1 x i64] }">
# CHECK:   %12 = llvm.insertvalue %7, %11[1] : !llvm<"{ i64*, i64*, i64, [1 x i64], [1 x i64] }">
# CHECK:   %13 = llvm.insertvalue %3, %12[2] : !llvm<"{ i64*, i64*, i64, [1 x i64], [1 x i64] }">
# CHECK:   %14 = llvm.getelementptr %9[%3] : (!llvm<"i64*">, !llvm.i64) -> !llvm<"i64*">
# CHECK:   %15 = llvm.load %14 : !llvm<"i64*">
# CHECK:   %16 = llvm.insertvalue %15, %13[3, 0] : !llvm<"{ i64*, i64*, i64, [1 x i64], [1 x i64] }">
# CHECK:   %17 = llvm.mlir.constant(1 : i64) : !llvm.i64
# CHECK:   %18 = llvm.insertvalue %17, %16[4, 0] : !llvm<"{ i64*, i64*, i64, [1 x i64], [1 x i64] }">
# CHECK:   %19 = llvm.extractvalue %18[1] : !llvm<"{ i64*, i64*, i64, [1 x i64], [1 x i64] }">
# CHECK:   %20 = llvm.mlir.constant(0 : index) : !llvm.i64
# CHECK:   %21 = llvm.mul %1, %0 : !llvm.i64
# CHECK:   %22 = llvm.add %20, %21 : !llvm.i64
# CHECK:   %23 = llvm.getelementptr %19[%22] : (!llvm<"i64*">, !llvm.i64) -> !llvm<"i64*">
# CHECK:   %24 = llvm.load %23 : !llvm<"i64*">
# CHECK:   llvm.return %24 : !llvm.i64

emit_lowered(index, Array{Int64, 3}, Int64)
# CHECK: #map0 = affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0 + d1 * s0 + d2 * s1)>
# CHECK: func @index(%arg0: !jlir<"typeof(Main.index)">, %arg1: !jlir<"Array{Int64, 3}">, %arg2: i64) -> i64
# CHECK:   %c0 = constant 0 : index
# CHECK:   %c1 = constant 1 : index
# CHECK:   %0 = "jlir.convertstd"(%arg2) : (i64) -> index
# CHECK:   %1 = subi %0, %c1 : index
# CHECK:   %2 = "jlir.arraytomemref"(%arg1) : (!jlir<"Array{Int64, 3}">) -> memref<?x?x?xi64, #map0>
# CHECK:   %3 = load %2[%c0, %c0, %1] : memref<?x?x?xi64, #map0>
# CHECK:   return %3 : i64
#
# CHECK: llvm.func @index(%arg0: !llvm<"%jl_value_t*">, %arg1: !llvm<"%jl_value_t*">, %arg2: !llvm.i64) -> !llvm.i64
# CHECK:   %0 = llvm.mlir.constant(0 : index) : !llvm.i64
# CHECK:   %1 = llvm.mlir.constant(1 : index) : !llvm.i64
# CHECK:   %2 = llvm.sub %arg2, %1 : !llvm.i64
# CHECK:   %3 = llvm.bitcast %arg1 : !llvm<"%jl_value_t*"> to !llvm<"%jl_array_t*">
# CHECK:   %4 = llvm.mlir.constant(0 : i64) : !llvm.i64
# CHECK:   %5 = llvm.mlir.constant(0 : i32) : !llvm.i32
# CHECK:   %6 = llvm.getelementptr %3[%4, %5] : (!llvm<"%jl_array_t*">, !llvm.i64, !llvm.i32) -> !llvm<"i64*">
# CHECK:   %7 = llvm.load %6 : !llvm<"i64*">
# CHECK:   %8 = llvm.bitcast %7 : !llvm<"i8*"> to !llvm<"i64*">
# CHECK:   %9 = llvm.mlir.constant(5 : i32) : !llvm.i32
# CHECK:   %10 = llvm.getelementptr %3[%4, %9] : (!llvm<"%jl_array_t*">, !llvm.i64, !llvm.i32) -> !llvm<"i64*">
# CHECK:   %11 = llvm.mlir.undef : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK:   %12 = llvm.insertvalue %8, %11[0] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK:   %13 = llvm.insertvalue %8, %12[1] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK:   %14 = llvm.insertvalue %4, %13[2] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK:   %15 = llvm.getelementptr %10[%4] : (!llvm<"i64*">, !llvm.i64) -> !llvm<"i64*">
# CHECK:   %16 = llvm.load %15 : !llvm<"i64*">
# CHECK:   %17 = llvm.insertvalue %16, %14[3, 2] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK:   %18 = llvm.mlir.constant(1 : i64) : !llvm.i64
# CHECK:   %19 = llvm.insertvalue %18, %17[4, 2] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK:   %20 = llvm.getelementptr %10[%18] : (!llvm<"i64*">, !llvm.i64) -> !llvm<"i64*">
# CHECK:   %21 = llvm.load %20 : !llvm<"i64*">
# CHECK:   %22 = llvm.insertvalue %21, %19[3, 1] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK:   %23 = llvm.mul %16, %18 : !llvm.i64
# CHECK:   %24 = llvm.insertvalue %23, %22[4, 1] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK:   %25 = llvm.mlir.constant(2 : i64) : !llvm.i64
# CHECK:   %26 = llvm.getelementptr %10[%25] : (!llvm<"i64*">, !llvm.i64) -> !llvm<"i64*">
# CHECK:   %27 = llvm.load %26 : !llvm<"i64*">
# CHECK:   %28 = llvm.insertvalue %27, %24[3, 0] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK:   %29 = llvm.mul %21, %23 : !llvm.i64
# CHECK:   %30 = llvm.insertvalue %29, %28[4, 0] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK:   %31 = llvm.extractvalue %30[1] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK:   %32 = llvm.mul %0, %1 : !llvm.i64
# CHECK:   %33 = llvm.add %0, %32 : !llvm.i64
# CHECK:   %34 = llvm.extractvalue %30[4, 1] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK:   %35 = llvm.mul %0, %34 : !llvm.i64
# CHECK:   %36 = llvm.add %33, %35 : !llvm.i64
# CHECK:   %37 = llvm.extractvalue %30[4, 2] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK:   %38 = llvm.mul %2, %37 : !llvm.i64
# CHECK:   %39 = llvm.add %36, %38 : !llvm.i64
# CHECK:   %40 = llvm.getelementptr %31[%39] : (!llvm<"i64*">, !llvm.i64) -> !llvm<"i64*">
# CHECK:   %41 = llvm.load %40 : !llvm<"i64*">
# CHECK:   llvm.return %41 : !llvm.i64

index(A, i, j, k) = A[i, j, k]
emit_lowered(index, Array{Int64, 3}, Int64, Int64, Int64)
# CHECK: #map0 = affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0 + d1 * s0 + d2 * s1)>
# CHECK: func @index(%arg0: !jlir<"typeof(Main.index)">, %arg1: !jlir<"Array{Int64, 3}">, %arg2: i64, %arg3: i64, %arg4: i64) -> i64
# CHECK:   %c1 = constant 1 : index
# CHECK:   %0 = "jlir.convertstd"(%arg2) : (i64) -> index
# CHECK:   %1 = subi %0, %c1 : index
# CHECK:   %2 = "jlir.convertstd"(%arg3) : (i64) -> index
# CHECK:   %3 = subi %2, %c1 : index
# CHECK:   %4 = "jlir.convertstd"(%arg4) : (i64) -> index
# CHECK:   %5 = subi %4, %c1 : index
# CHECK:   %6 = "jlir.arraytomemref"(%arg1) : (!jlir<"Array{Int64, 3}">) -> memref<?x?x?xi64, #map0>
# CHECK:   %7 = load %6[%1, %3, %5] : memref<?x?x?xi64, #map0>
# CHECK:   return %7 : i64
#
# CHECK: llvm.func @index(%arg0: !llvm<"%jl_value_t*">, %arg1: !llvm<"%jl_value_t*">, %arg2: !llvm.i64, %arg3: !llvm.i64, %arg4: !llvm.i64) -> !llvm.i64
# CHECK:   %0 = llvm.mlir.constant(1 : index) : !llvm.i64
# CHECK:   %1 = llvm.sub %arg2, %0 : !llvm.i64
# CHECK:   %2 = llvm.sub %arg3, %0 : !llvm.i64
# CHECK:   %3 = llvm.sub %arg4, %0 : !llvm.i64
# CHECK:   %4 = llvm.bitcast %arg1 : !llvm<"%jl_value_t*"> to !llvm<"%jl_array_t*">
# CHECK:   %5 = llvm.mlir.constant(0 : i64) : !llvm.i64
# CHECK:   %6 = llvm.mlir.constant(0 : i32) : !llvm.i32
# CHECK:   %7 = llvm.getelementptr %4[%5, %6] : (!llvm<"%jl_array_t*">, !llvm.i64, !llvm.i32) -> !llvm<"i64*">
# CHECK:   %8 = llvm.load %7 : !llvm<"i64*">
# CHECK:   %9 = llvm.bitcast %8 : !llvm<"i8*"> to !llvm<"i64*">
# CHECK:   %10 = llvm.mlir.constant(5 : i32) : !llvm.i32
# CHECK:   %11 = llvm.getelementptr %4[%5, %10] : (!llvm<"%jl_array_t*">, !llvm.i64, !llvm.i32) -> !llvm<"i64*">
# CHECK:   %12 = llvm.mlir.undef : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK:   %13 = llvm.insertvalue %9, %12[0] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK:   %14 = llvm.insertvalue %9, %13[1] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK:   %15 = llvm.insertvalue %5, %14[2] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK:   %16 = llvm.getelementptr %11[%5] : (!llvm<"i64*">, !llvm.i64) -> !llvm<"i64*">
# CHECK:   %17 = llvm.load %16 : !llvm<"i64*">
# CHECK:   %18 = llvm.insertvalue %17, %15[3, 2] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK:   %19 = llvm.mlir.constant(1 : i64) : !llvm.i64
# CHECK:   %20 = llvm.insertvalue %19, %18[4, 2] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK:   %21 = llvm.getelementptr %11[%19] : (!llvm<"i64*">, !llvm.i64) -> !llvm<"i64*">
# CHECK:   %22 = llvm.load %21 : !llvm<"i64*">
# CHECK:   %23 = llvm.insertvalue %22, %20[3, 1] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK:   %24 = llvm.mul %17, %19 : !llvm.i64
# CHECK:   %25 = llvm.insertvalue %24, %23[4, 1] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK:   %26 = llvm.mlir.constant(2 : i64) : !llvm.i64
# CHECK:   %27 = llvm.getelementptr %11[%26] : (!llvm<"i64*">, !llvm.i64) -> !llvm<"i64*">
# CHECK:   %28 = llvm.load %27 : !llvm<"i64*">
# CHECK:   %29 = llvm.insertvalue %28, %25[3, 0] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK:   %30 = llvm.mul %22, %24 : !llvm.i64
# CHECK:   %31 = llvm.insertvalue %30, %29[4, 0] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK:   %32 = llvm.extractvalue %31[1] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK:   %33 = llvm.mlir.constant(0 : index) : !llvm.i64
# CHECK:   %34 = llvm.mul %1, %0 : !llvm.i64
# CHECK:   %35 = llvm.add %33, %34 : !llvm.i64
# CHECK:   %36 = llvm.extractvalue %31[4, 1] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK:   %37 = llvm.mul %2, %36 : !llvm.i64
# CHECK:   %38 = llvm.add %35, %37 : !llvm.i64
# CHECK:   %39 = llvm.extractvalue %31[4, 2] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK:   %40 = llvm.mul %3, %39 : !llvm.i64
# CHECK:   %41 = llvm.add %38, %40 : !llvm.i64
# CHECK:   %42 = llvm.getelementptr %32[%41] : (!llvm<"i64*">, !llvm.i64) -> !llvm<"i64*">
# CHECK:   %43 = llvm.load %42 : !llvm<"i64*">
# CHECK:   llvm.return %43 : !llvm.i64
