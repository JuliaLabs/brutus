# RUN: julia -e "import Brutus; Brutus.lit(:emit_lowered)" --startup-file=no %s 2>&1 | FileCheck %s

index(A, i, j, k) = A[i, j, k]
emit(index, Array{Int64, 3}, Int64, Int64, Int64)
#



# CHECK: module {
# CHECK-NEXT:   func @"Tuple{typeof(Main.index), Array{Int64, 3}, Int64, Int64, Int64}"(%arg0: !jlir<"typeof(Main.index)">, %arg1: !jlir<"Array{Int64, 3}">, %arg2: i64, %arg3: i64, %arg4: i64) -> i64 {
# CHECK-NEXT:     %c1 = constant 1 : index
# CHECK-NEXT:     %0 = "jlir.arraytomemref"(%arg1) : (!jlir<"Array{Int64, 3}">) -> memref<?x?x?xi64>
# CHECK-NEXT:     %1 = "jlir.convertstd"(%arg2) : (i64) -> index
# CHECK-NEXT:     %2 = subi %1, %c1 : index
# CHECK-NEXT:     %3 = "jlir.convertstd"(%arg3) : (i64) -> index
# CHECK-NEXT:     %4 = subi %3, %c1 : index
# CHECK-NEXT:     %5 = "jlir.convertstd"(%arg4) : (i64) -> index
# CHECK-NEXT:     %6 = subi %5, %c1 : index
# CHECK-NEXT:     %7 = load %0[%6, %4, %2] : memref<?x?x?xi64>
# CHECK-NEXT:     return %7 : i64
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK: module {
# CHECK-NEXT:   llvm.func @"Tuple{typeof(Main.index), Array{Int64, 3}, Int64, Int64, Int64}"(%arg0: !llvm<"%jl_value_t*">, %arg1: !llvm<"%jl_value_t*">, %arg2: !llvm.i64, %arg3: !llvm.i64, %arg4: !llvm.i64) -> !llvm.i64 {
# CHECK-NEXT:     %0 = llvm.mlir.constant(1 : index) : !llvm.i64
# CHECK-NEXT:     %1 = llvm.bitcast %arg1 : !llvm<"%jl_value_t*"> to !llvm<"%jl_array_t*">
# CHECK-NEXT:     %2 = llvm.mlir.constant({{[0-9]+}} : i64) : !llvm.i64
# CHECK-NEXT:     %3 = llvm.mlir.constant(0 : i32) : !llvm.i32
# CHECK-NEXT:     %4 = llvm.getelementptr %1[%2, %3] : (!llvm<"%jl_array_t*">, !llvm.i64, !llvm.i32) -> !llvm<"i64*">
# CHECK-NEXT:     %5 = llvm.load %4 : !llvm<"i64*">
# CHECK-NEXT:     %6 = llvm.bitcast %5 : !llvm<"i8*"> to !llvm<"i64*">
# CHECK-NEXT:     %7 = llvm.mlir.constant(5 : i32) : !llvm.i32
# CHECK-NEXT:     %8 = llvm.getelementptr %1[%2, %7] : (!llvm<"%jl_array_t*">, !llvm.i64, !llvm.i32) -> !llvm<"i64*">
# CHECK-NEXT:     %9 = llvm.mlir.undef : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %10 = llvm.insertvalue %6, %9[0] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %11 = llvm.insertvalue %6, %10[1] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %12 = llvm.insertvalue %2, %11[2] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %13 = llvm.getelementptr %8[%2] : (!llvm<"i64*">, !llvm.i64) -> !llvm<"i64*">
# CHECK-NEXT:     %14 = llvm.load %13 : !llvm<"i64*">
# CHECK-NEXT:     %15 = llvm.insertvalue %14, %12[3, 2] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %16 = llvm.mlir.constant({{[0-9]+}} : i64) : !llvm.i64
# CHECK-NEXT:     %17 = llvm.insertvalue %16, %15[4, 2] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %18 = llvm.getelementptr %8[%16] : (!llvm<"i64*">, !llvm.i64) -> !llvm<"i64*">
# CHECK-NEXT:     %19 = llvm.load %18 : !llvm<"i64*">
# CHECK-NEXT:     %20 = llvm.insertvalue %19, %17[3, 1] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %21 = llvm.mul %14, %16 : !llvm.i64
# CHECK-NEXT:     %22 = llvm.insertvalue %21, %20[4, 1] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %23 = llvm.mlir.constant({{[0-9]+}} : i64) : !llvm.i64
# CHECK-NEXT:     %24 = llvm.getelementptr %8[%23] : (!llvm<"i64*">, !llvm.i64) -> !llvm<"i64*">
# CHECK-NEXT:     %25 = llvm.load %24 : !llvm<"i64*">
# CHECK-NEXT:     %26 = llvm.insertvalue %25, %22[3, 0] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %27 = llvm.mul %19, %21 : !llvm.i64
# CHECK-NEXT:     %28 = llvm.insertvalue %27, %26[4, 0] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %29 = llvm.sub %arg2, %0 : !llvm.i64
# CHECK-NEXT:     %30 = llvm.sub %arg3, %0 : !llvm.i64
# CHECK-NEXT:     %31 = llvm.sub %arg4, %0 : !llvm.i64
# CHECK-NEXT:     %32 = llvm.extractvalue %28[1] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %33 = llvm.mlir.constant(0 : index) : !llvm.i64
# CHECK-NEXT:     %34 = llvm.extractvalue %28[4, 0] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %35 = llvm.mul %31, %34 : !llvm.i64
# CHECK-NEXT:     %36 = llvm.add %33, %35 : !llvm.i64
# CHECK-NEXT:     %37 = llvm.extractvalue %28[4, 1] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %38 = llvm.mul %30, %37 : !llvm.i64
# CHECK-NEXT:     %39 = llvm.add %36, %38 : !llvm.i64
# CHECK-NEXT:     %40 = llvm.mul %29, %0 : !llvm.i64
# CHECK-NEXT:     %41 = llvm.add %39, %40 : !llvm.i64
# CHECK-NEXT:     %42 = llvm.getelementptr %32[%41] : (!llvm<"i64*">, !llvm.i64) -> !llvm<"i64*">
# CHECK-NEXT:     %43 = llvm.load %42 : !llvm<"i64*">
# CHECK-NEXT:     llvm.return %43 : !llvm.i64
# CHECK-NEXT:   }
# CHECK-NEXT: }
