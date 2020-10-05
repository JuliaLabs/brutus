# RUN: julia -e "import Brutus; Brutus.lit(:emit_lowered)" --startup-file=no %s 2>&1 | FileCheck %s

index(A, i) = A[i]
emit(index, Array{Int64, 3}, Int64)
# CHECK: func @"Tuple{typeof(Main.index), Array{Int64, 3}, Int64}"(%arg0: !jlir<"typeof(Main.index)">, %arg1: !jlir<"Array{Int64, 3}">, %arg2: i64) -> i64
# CHECK:   %c0 = constant 0 : index
# CHECK:   %c1 = constant 1 : index
# CHECK:   %0 = "jlir.convertstd"(%arg2) : (i64) -> index
# CHECK:   %1 = subi %0, %c1 : index
# CHECK:   %2 = "jlir.arraytomemref"(%arg1) : (!jlir<"Array{Int64, 3}">) -> memref<?x?x?xi64>
# CHECK:   %3 = load %2[%c0, %c0, %1] : memref<?x?x?xi64>
# CHECK:   return %3 : i64
#
# CHECK: llvm.func @"Tuple{typeof(Main.index), Array{Int64, 3}, Int64}"(%arg0: !llvm<"%jl_value_t*">, %arg1: !llvm<"%jl_value_t*">, %arg2: !llvm.i64) -> !llvm.i64
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
# CHECK:   %32 = llvm.extractvalue %30[4, 0] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK:   %33 = llvm.mul %0, %32 : !llvm.i64
# CHECK:   %34 = llvm.add %0, %33 : !llvm.i64
# CHECK:   %35 = llvm.extractvalue %30[4, 1] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK:   %36 = llvm.mul %0, %35 : !llvm.i64
# CHECK:   %37 = llvm.add %34, %36 : !llvm.i64
# CHECK:   %38 = llvm.mul %2, %1 : !llvm.i64
# CHECK:   %39 = llvm.add %37, %38 : !llvm.i64
# CHECK:   %40 = llvm.getelementptr %31[%39] : (!llvm<"i64*">, !llvm.i64) -> !llvm<"i64*">
# CHECK:   %41 = llvm.load %40 : !llvm<"i64*">
# CHECK:   llvm.return %41 : !llvm.i64
