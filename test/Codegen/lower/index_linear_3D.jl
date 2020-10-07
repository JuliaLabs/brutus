# RUN: julia -e "import Brutus; Brutus.lit(:emit_lowered)" --startup-file=no %s 2>&1 | FileCheck %s

index(A, i) = A[i]
emit(index, Array{Int64, 3}, Int64)
#



# CHECK: module {
# CHECK-NEXT:   func @"Tuple{typeof(Main.index), Array{Int64, 3}, Int64}"(%arg0: !jlir<"typeof(Main.index)">, %arg1: !jlir<"Array{Int64, 3}">, %arg2: i64) -> i64 {
# CHECK-NEXT:     %c0 = constant 0 : index
# CHECK-NEXT:     %c1 = constant 1 : index
# CHECK-NEXT:     %0 = "jlir.convertstd"(%arg2) : (i64) -> index
# CHECK-NEXT:     %1 = subi %0, %c1 : index
# CHECK-NEXT:     %2 = "jlir.arraytomemref"(%arg1) : (!jlir<"Array{Int64, 3}">) -> memref<?x?x?xi64>
# CHECK-NEXT:     %3 = load %2[%c0, %c0, %1] : memref<?x?x?xi64>
# CHECK-NEXT:     return %3 : i64
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK: module {
# CHECK-NEXT:   llvm.func @"Tuple{typeof(Main.index), Array{Int64, 3}, Int64}"(%arg0: !llvm<"%jl_value_t*">, %arg1: !llvm<"%jl_value_t*">, %arg2: !llvm.i64) -> !llvm.i64 {
# CHECK-NEXT:     %0 = llvm.mlir.constant(0 : index) : !llvm.i64
# CHECK-NEXT:     %1 = llvm.mlir.constant(1 : index) : !llvm.i64
# CHECK-NEXT:     %2 = llvm.sub %arg2, %1 : !llvm.i64
# CHECK-NEXT:     %3 = llvm.bitcast %arg1 : !llvm<"%jl_value_t*"> to !llvm<"%jl_array_t*">
# CHECK-NEXT:     %4 = llvm.mlir.constant({{[0-9]+}} : i64) : !llvm.i64
# CHECK-NEXT:     %5 = llvm.mlir.constant(0 : i32) : !llvm.i32
# CHECK-NEXT:     %6 = llvm.getelementptr %3[%4, %5] : (!llvm<"%jl_array_t*">, !llvm.i64, !llvm.i32) -> !llvm<"i64*">
# CHECK-NEXT:     %7 = llvm.load %6 : !llvm<"i64*">
# CHECK-NEXT:     %8 = llvm.bitcast %7 : !llvm<"i8*"> to !llvm<"i64*">
# CHECK-NEXT:     %9 = llvm.mlir.constant(5 : i32) : !llvm.i32
# CHECK-NEXT:     %10 = llvm.getelementptr %3[%4, %9] : (!llvm<"%jl_array_t*">, !llvm.i64, !llvm.i32) -> !llvm<"i64*">
# CHECK-NEXT:     %11 = llvm.mlir.undef : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %12 = llvm.insertvalue %8, %11[0] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %13 = llvm.insertvalue %8, %12[1] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %14 = llvm.insertvalue %4, %13[2] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %15 = llvm.getelementptr %10[%4] : (!llvm<"i64*">, !llvm.i64) -> !llvm<"i64*">
# CHECK-NEXT:     %16 = llvm.load %15 : !llvm<"i64*">
# CHECK-NEXT:     %17 = llvm.insertvalue %16, %14[3, 2] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %18 = llvm.mlir.constant({{[0-9]+}} : i64) : !llvm.i64
# CHECK-NEXT:     %19 = llvm.insertvalue %18, %17[4, 2] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %20 = llvm.getelementptr %10[%18] : (!llvm<"i64*">, !llvm.i64) -> !llvm<"i64*">
# CHECK-NEXT:     %21 = llvm.load %20 : !llvm<"i64*">
# CHECK-NEXT:     %22 = llvm.insertvalue %21, %19[3, 1] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %23 = llvm.mul %16, %18 : !llvm.i64
# CHECK-NEXT:     %24 = llvm.insertvalue %23, %22[4, 1] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %25 = llvm.mlir.constant({{[0-9]+}} : i64) : !llvm.i64
# CHECK-NEXT:     %26 = llvm.getelementptr %10[%25] : (!llvm<"i64*">, !llvm.i64) -> !llvm<"i64*">
# CHECK-NEXT:     %27 = llvm.load %26 : !llvm<"i64*">
# CHECK-NEXT:     %28 = llvm.insertvalue %27, %24[3, 0] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %29 = llvm.mul %21, %23 : !llvm.i64
# CHECK-NEXT:     %30 = llvm.insertvalue %29, %28[4, 0] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %31 = llvm.extractvalue %30[1] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %32 = llvm.extractvalue %30[4, 0] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %33 = llvm.mul %0, %32 : !llvm.i64
# CHECK-NEXT:     %34 = llvm.add %0, %33 : !llvm.i64
# CHECK-NEXT:     %35 = llvm.extractvalue %30[4, 1] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %36 = llvm.mul %0, %35 : !llvm.i64
# CHECK-NEXT:     %37 = llvm.add %34, %36 : !llvm.i64
# CHECK-NEXT:     %38 = llvm.mul %2, %1 : !llvm.i64
# CHECK-NEXT:     %39 = llvm.add %37, %38 : !llvm.i64
# CHECK-NEXT:     %40 = llvm.getelementptr %31[%39] : (!llvm<"i64*">, !llvm.i64) -> !llvm<"i64*">
# CHECK-NEXT:     %41 = llvm.load %40 : !llvm<"i64*">
# CHECK-NEXT:     llvm.return %41 : !llvm.i64
# CHECK-NEXT:   }
# CHECK-NEXT: }
