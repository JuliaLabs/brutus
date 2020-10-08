# RUN: julia -e "import Brutus; Brutus.lit(:emit_lowered)" --startup-file=no %s 2>&1 | FileCheck %s

index(A, i) = A[i]
emit(index, Array{Int64, 3}, Int64)
#



# CHECK: module {
# CHECK-NEXT:   func @"Tuple{typeof(Main.index), Array{Int64, 3}, Int64}"(%arg0: !jlir<"typeof(Main.index)">, %arg1: !jlir<"Array{Int64, 3}">, %arg2: i64) -> i64 {
# CHECK-NEXT:     %c0 = constant 0 : index
# CHECK-NEXT:     %c1 = constant 1 : index
# CHECK-NEXT:     %0 = "jlir.arraytomemref"(%arg1) : (!jlir<"Array{Int64, 3}">) -> memref<?x?x?xi64>
# CHECK-NEXT:     %1 = "jlir.convertstd"(%arg2) : (i64) -> index
# CHECK-NEXT:     %2 = subi %1, %c1 : index
# CHECK-NEXT:     %3 = load %0[%c0, %c0, %2] : memref<?x?x?xi64>
# CHECK-NEXT:     return %3 : i64
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK: module {
# CHECK-NEXT:   llvm.func @"Tuple{typeof(Main.index), Array{Int64, 3}, Int64}"(%arg0: !llvm<"%jl_value_t*">, %arg1: !llvm<"%jl_value_t*">, %arg2: !llvm.i64) -> !llvm.i64 {
# CHECK-NEXT:     %0 = llvm.mlir.constant(0 : index) : !llvm.i64
# CHECK-NEXT:     %1 = llvm.mlir.constant(1 : index) : !llvm.i64
# CHECK-NEXT:     %2 = llvm.bitcast %arg1 : !llvm<"%jl_value_t*"> to !llvm<"%jl_array_t*">
# CHECK-NEXT:     %3 = llvm.mlir.constant({{[0-9]+}} : i64) : !llvm.i64
# CHECK-NEXT:     %4 = llvm.mlir.constant(0 : i32) : !llvm.i32
# CHECK-NEXT:     %5 = llvm.getelementptr %2[%3, %4] : (!llvm<"%jl_array_t*">, !llvm.i64, !llvm.i32) -> !llvm<"i64*">
# CHECK-NEXT:     %6 = llvm.load %5 : !llvm<"i64*">
# CHECK-NEXT:     %7 = llvm.bitcast %6 : !llvm<"i8*"> to !llvm<"i64*">
# CHECK-NEXT:     %8 = llvm.mlir.constant(5 : i32) : !llvm.i32
# CHECK-NEXT:     %9 = llvm.getelementptr %2[%3, %8] : (!llvm<"%jl_array_t*">, !llvm.i64, !llvm.i32) -> !llvm<"i64*">
# CHECK-NEXT:     %10 = llvm.mlir.undef : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %11 = llvm.insertvalue %7, %10[0] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %12 = llvm.insertvalue %7, %11[1] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %13 = llvm.insertvalue %3, %12[2] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %14 = llvm.getelementptr %9[%3] : (!llvm<"i64*">, !llvm.i64) -> !llvm<"i64*">
# CHECK-NEXT:     %15 = llvm.load %14 : !llvm<"i64*">
# CHECK-NEXT:     %16 = llvm.insertvalue %15, %13[3, 2] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %17 = llvm.mlir.constant({{[0-9]+}} : i64) : !llvm.i64
# CHECK-NEXT:     %18 = llvm.insertvalue %17, %16[4, 2] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %19 = llvm.getelementptr %9[%17] : (!llvm<"i64*">, !llvm.i64) -> !llvm<"i64*">
# CHECK-NEXT:     %20 = llvm.load %19 : !llvm<"i64*">
# CHECK-NEXT:     %21 = llvm.insertvalue %20, %18[3, 1] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %22 = llvm.mul %15, %17 : !llvm.i64
# CHECK-NEXT:     %23 = llvm.insertvalue %22, %21[4, 1] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %24 = llvm.mlir.constant({{[0-9]+}} : i64) : !llvm.i64
# CHECK-NEXT:     %25 = llvm.getelementptr %9[%24] : (!llvm<"i64*">, !llvm.i64) -> !llvm<"i64*">
# CHECK-NEXT:     %26 = llvm.load %25 : !llvm<"i64*">
# CHECK-NEXT:     %27 = llvm.insertvalue %26, %23[3, 0] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %28 = llvm.mul %20, %22 : !llvm.i64
# CHECK-NEXT:     %29 = llvm.insertvalue %28, %27[4, 0] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %30 = llvm.sub %arg2, %1 : !llvm.i64
# CHECK-NEXT:     %31 = llvm.extractvalue %29[1] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %32 = llvm.extractvalue %29[4, 0] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %33 = llvm.mul %0, %32 : !llvm.i64
# CHECK-NEXT:     %34 = llvm.add %0, %33 : !llvm.i64
# CHECK-NEXT:     %35 = llvm.extractvalue %29[4, 1] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %36 = llvm.mul %0, %35 : !llvm.i64
# CHECK-NEXT:     %37 = llvm.add %34, %36 : !llvm.i64
# CHECK-NEXT:     %38 = llvm.mul %30, %1 : !llvm.i64
# CHECK-NEXT:     %39 = llvm.add %37, %38 : !llvm.i64
# CHECK-NEXT:     %40 = llvm.getelementptr %31[%39] : (!llvm<"i64*">, !llvm.i64) -> !llvm<"i64*">
# CHECK-NEXT:     %41 = llvm.load %40 : !llvm<"i64*">
# CHECK-NEXT:     llvm.return %41 : !llvm.i64
# CHECK-NEXT:   }
# CHECK-NEXT: }
