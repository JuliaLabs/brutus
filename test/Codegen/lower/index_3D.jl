# RUN: julia -e "import Brutus; Brutus.lit(:emit_lowered)" --startup-file=no %s 2>&1 | FileCheck %s

index(A, i, j, k) = A[i, j, k]
emit(index, Array{Int64, 3}, Int64, Int64, Int64)
# CHECK: func @"Tuple{typeof(Main.index), Array{Int64, 3}, Int64, Int64, Int64}"(%arg0: !jlir<"typeof(Main.index)">, %arg1: !jlir<"Array{Int64, 3}">, %arg2: i64, %arg3: i64, %arg4: i64) -> i64
# CHECK:   %c1 = constant 1 : index
# CHECK:   %0 = "jlir.convertstd"(%arg2) : (i64) -> index
# CHECK:   %1 = subi %0, %c1 : index
# CHECK:   %2 = "jlir.convertstd"(%arg3) : (i64) -> index
# CHECK:   %3 = subi %2, %c1 : index
# CHECK:   %4 = "jlir.convertstd"(%arg4) : (i64) -> index
# CHECK:   %5 = subi %4, %c1 : index
# CHECK:   %6 = "jlir.arraytomemref"(%arg1) : (!jlir<"Array{Int64, 3}">) -> memref<?x?x?xi64>
# CHECK:   %7 = load %6[%5, %3, %1] : memref<?x?x?xi64>
# CHECK:   return %7 : i64
#
# CHECK: llvm.func @"Tuple{typeof(Main.index), Array{Int64, 3}, Int64, Int64, Int64}"(%arg0: !llvm<"%jl_value_t*">, %arg1: !llvm<"%jl_value_t*">, %arg2: !llvm.i64, %arg3: !llvm.i64, %arg4: !llvm.i64) -> !llvm.i64
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
# CHECK:   %34 = llvm.extractvalue %31[4, 0] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK:   %35 = llvm.mul %3, %34 : !llvm.i64
# CHECK:   %36 = llvm.add %33, %35 : !llvm.i64
# CHECK:   %37 = llvm.extractvalue %31[4, 1] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK:   %38 = llvm.mul %2, %37 : !llvm.i64
# CHECK:   %39 = llvm.add %36, %38 : !llvm.i64
# CHECK:   %40 = llvm.mul %1, %0 : !llvm.i64
# CHECK:   %41 = llvm.add %39, %40 : !llvm.i64
# CHECK:   %42 = llvm.getelementptr %32[%41] : (!llvm<"i64*">, !llvm.i64) -> !llvm<"i64*">
# CHECK:   %43 = llvm.load %42 : !llvm<"i64*">
# CHECK:   llvm.return %43 : !llvm.i64
