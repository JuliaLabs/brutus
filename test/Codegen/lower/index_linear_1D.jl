# RUN: julia -e "import Brutus; Brutus.lit(:emit_lowered)" --startup-file=no %s 2>&1 | FileCheck %s

index(A, i) = A[i]
emit(index, Array{Int64, 1}, Int64)
# CHECK: func @"Tuple{typeof(Main.index), Array{Int64, 1}, Int64}"(%arg0: !jlir<"typeof(Main.index)">, %arg1: !jlir<"Array{Int64, 1}">, %arg2: i64) -> i64
# CHECK:   %c1 = constant 1 : index
# CHECK:   %0 = "jlir.convertstd"(%arg2) : (i64) -> index
# CHECK:   %1 = subi %0, %c1 : index
# CHECK:   %2 = "jlir.arraytomemref"(%arg1) : (!jlir<"Array{Int64, 1}">) -> memref<?xi64>
# CHECK:   %3 = load %2[%1] : memref<?xi64>
# CHECK:   return %3 : i64
#
# CHECK: llvm.func @"Tuple{typeof(Main.index), Array{Int64, 1}, Int64}"(%arg0: !llvm<"%jl_value_t*">, %arg1: !llvm<"%jl_value_t*">, %arg2: !llvm.i64) -> !llvm.i64
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
