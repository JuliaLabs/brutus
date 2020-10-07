# RUN: julia -e "import Brutus; Brutus.lit(:emit_lowered)" --startup-file=no %s 2>&1 | FileCheck %s

index(A, i, j, k) = A[i, j, k]
emit(index, Array{Int64, 3}, Int64, Int64, Int64)
#



# CHECK: module {
# CHECK-NEXT:   func @"Tuple{typeof(Main.index), Array{Int64, 3}, Int64, Int64, Int64}"(%arg0: !jlir<"typeof(Main.index)">, %arg1: !jlir<"Array{Int64, 3}">, %arg2: i64, %arg3: i64, %arg4: i64) -> i64 {
# CHECK-NEXT:     %c1 = constant 1 : index
# CHECK-NEXT:     %0 = "jlir.convertstd"(%arg2) : (i64) -> index
# CHECK-NEXT:     %1 = subi %0, %c1 : index
# CHECK-NEXT:     %2 = "jlir.convertstd"(%arg3) : (i64) -> index
# CHECK-NEXT:     %3 = subi %2, %c1 : index
# CHECK-NEXT:     %4 = "jlir.convertstd"(%arg4) : (i64) -> index
# CHECK-NEXT:     %5 = subi %4, %c1 : index
# CHECK-NEXT:     %6 = "jlir.arraytomemref"(%arg1) : (!jlir<"Array{Int64, 3}">) -> memref<?x?x?xi64>
# CHECK-NEXT:     %7 = load %6[%5, %3, %1] : memref<?x?x?xi64>
# CHECK-NEXT:     return %7 : i64
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK: module {
# CHECK-NEXT:   llvm.func @"Tuple{typeof(Main.index), Array{Int64, 3}, Int64, Int64, Int64}"(%arg0: !llvm<"%jl_value_t*">, %arg1: !llvm<"%jl_value_t*">, %arg2: !llvm.i64, %arg3: !llvm.i64, %arg4: !llvm.i64) -> !llvm.i64 {
# CHECK-NEXT:     %0 = llvm.mlir.constant(1 : index) : !llvm.i64
# CHECK-NEXT:     %1 = llvm.sub %arg2, %0 : !llvm.i64
# CHECK-NEXT:     %2 = llvm.sub %arg3, %0 : !llvm.i64
# CHECK-NEXT:     %3 = llvm.sub %arg4, %0 : !llvm.i64
# CHECK-NEXT:     %4 = llvm.bitcast %arg1 : !llvm<"%jl_value_t*"> to !llvm<"%jl_array_t*">
# CHECK-NEXT:     %5 = llvm.mlir.constant({{[0-9]+}} : i64) : !llvm.i64
# CHECK-NEXT:     %6 = llvm.mlir.constant(0 : i32) : !llvm.i32
# CHECK-NEXT:     %7 = llvm.getelementptr %4[%5, %6] : (!llvm<"%jl_array_t*">, !llvm.i64, !llvm.i32) -> !llvm<"i64*">
# CHECK-NEXT:     %8 = llvm.load %7 : !llvm<"i64*">
# CHECK-NEXT:     %9 = llvm.bitcast %8 : !llvm<"i8*"> to !llvm<"i64*">
# CHECK-NEXT:     %10 = llvm.mlir.constant(5 : i32) : !llvm.i32
# CHECK-NEXT:     %11 = llvm.getelementptr %4[%5, %10] : (!llvm<"%jl_array_t*">, !llvm.i64, !llvm.i32) -> !llvm<"i64*">
# CHECK-NEXT:     %12 = llvm.mlir.undef : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %13 = llvm.insertvalue %9, %12[0] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %14 = llvm.insertvalue %9, %13[1] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %15 = llvm.insertvalue %5, %14[2] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %16 = llvm.getelementptr %11[%5] : (!llvm<"i64*">, !llvm.i64) -> !llvm<"i64*">
# CHECK-NEXT:     %17 = llvm.load %16 : !llvm<"i64*">
# CHECK-NEXT:     %18 = llvm.insertvalue %17, %15[3, 2] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %19 = llvm.mlir.constant({{[0-9]+}} : i64) : !llvm.i64
# CHECK-NEXT:     %20 = llvm.insertvalue %19, %18[4, 2] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %21 = llvm.getelementptr %11[%19] : (!llvm<"i64*">, !llvm.i64) -> !llvm<"i64*">
# CHECK-NEXT:     %22 = llvm.load %21 : !llvm<"i64*">
# CHECK-NEXT:     %23 = llvm.insertvalue %22, %20[3, 1] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %24 = llvm.mul %17, %19 : !llvm.i64
# CHECK-NEXT:     %25 = llvm.insertvalue %24, %23[4, 1] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %26 = llvm.mlir.constant({{[0-9]+}} : i64) : !llvm.i64
# CHECK-NEXT:     %27 = llvm.getelementptr %11[%26] : (!llvm<"i64*">, !llvm.i64) -> !llvm<"i64*">
# CHECK-NEXT:     %28 = llvm.load %27 : !llvm<"i64*">
# CHECK-NEXT:     %29 = llvm.insertvalue %28, %25[3, 0] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %30 = llvm.mul %22, %24 : !llvm.i64
# CHECK-NEXT:     %31 = llvm.insertvalue %30, %29[4, 0] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %32 = llvm.extractvalue %31[1] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %33 = llvm.mlir.constant(0 : index) : !llvm.i64
# CHECK-NEXT:     %34 = llvm.extractvalue %31[4, 0] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %35 = llvm.mul %3, %34 : !llvm.i64
# CHECK-NEXT:     %36 = llvm.add %33, %35 : !llvm.i64
# CHECK-NEXT:     %37 = llvm.extractvalue %31[4, 1] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %38 = llvm.mul %2, %37 : !llvm.i64
# CHECK-NEXT:     %39 = llvm.add %36, %38 : !llvm.i64
# CHECK-NEXT:     %40 = llvm.mul %1, %0 : !llvm.i64
# CHECK-NEXT:     %41 = llvm.add %39, %40 : !llvm.i64
# CHECK-NEXT:     %42 = llvm.getelementptr %32[%41] : (!llvm<"i64*">, !llvm.i64) -> !llvm<"i64*">
# CHECK-NEXT:     %43 = llvm.load %42 : !llvm<"i64*">
# CHECK-NEXT:     llvm.return %43 : !llvm.i64
# CHECK-NEXT:   }
# CHECK-NEXT: }
