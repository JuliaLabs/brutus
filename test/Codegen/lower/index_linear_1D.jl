# RUN: julia -e "import Brutus; Brutus.lit(:emit_lowered)" --startup-file=no %s 2>&1 | FileCheck %s

index(A, i) = A[i]
emit(index, Array{Int64, 1}, Int64)
#



# CHECK: module {
# CHECK-NEXT:   func @"Tuple{typeof(Main.index), Array{Int64, 1}, Int64}"(%arg0: !jlir<"typeof(Main.index)">, %arg1: !jlir<"Array{Int64, 1}">, %arg2: i64) -> i64 {
# CHECK-NEXT:     %c1 = constant 1 : index
# CHECK-NEXT:     %0 = "jlir.convertstd"(%arg2) : (i64) -> index
# CHECK-NEXT:     %1 = subi %0, %c1 : index
# CHECK-NEXT:     %2 = "jlir.arraytomemref"(%arg1) : (!jlir<"Array{Int64, 1}">) -> memref<?xi64>
# CHECK-NEXT:     %3 = load %2[%1] : memref<?xi64>
# CHECK-NEXT:     return %3 : i64
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK: module {
# CHECK-NEXT:   llvm.func @"Tuple{typeof(Main.index), Array{Int64, 1}, Int64}"(%arg0: !llvm<"%jl_value_t*">, %arg1: !llvm<"%jl_value_t*">, %arg2: !llvm.i64) -> !llvm.i64 {
# CHECK-NEXT:     %0 = llvm.mlir.constant(1 : index) : !llvm.i64
# CHECK-NEXT:     %1 = llvm.sub %arg2, %0 : !llvm.i64
# CHECK-NEXT:     %2 = llvm.bitcast %arg1 : !llvm<"%jl_value_t*"> to !llvm<"%jl_array_t*">
# CHECK-NEXT:     %3 = llvm.mlir.constant({{[0-9]+}} : i64) : !llvm.i64
# CHECK-NEXT:     %4 = llvm.mlir.constant(0 : i32) : !llvm.i32
# CHECK-NEXT:     %5 = llvm.getelementptr %2[%3, %4] : (!llvm<"%jl_array_t*">, !llvm.i64, !llvm.i32) -> !llvm<"i64*">
# CHECK-NEXT:     %6 = llvm.load %5 : !llvm<"i64*">
# CHECK-NEXT:     %7 = llvm.bitcast %6 : !llvm<"i8*"> to !llvm<"i64*">
# CHECK-NEXT:     %8 = llvm.mlir.constant(5 : i32) : !llvm.i32
# CHECK-NEXT:     %9 = llvm.getelementptr %2[%3, %8] : (!llvm<"%jl_array_t*">, !llvm.i64, !llvm.i32) -> !llvm<"i64*">
# CHECK-NEXT:     %10 = llvm.mlir.undef : !llvm<"{ i64*, i64*, i64, [1 x i64], [1 x i64] }">
# CHECK-NEXT:     %11 = llvm.insertvalue %7, %10[0] : !llvm<"{ i64*, i64*, i64, [1 x i64], [1 x i64] }">
# CHECK-NEXT:     %12 = llvm.insertvalue %7, %11[1] : !llvm<"{ i64*, i64*, i64, [1 x i64], [1 x i64] }">
# CHECK-NEXT:     %13 = llvm.insertvalue %3, %12[2] : !llvm<"{ i64*, i64*, i64, [1 x i64], [1 x i64] }">
# CHECK-NEXT:     %14 = llvm.getelementptr %9[%3] : (!llvm<"i64*">, !llvm.i64) -> !llvm<"i64*">
# CHECK-NEXT:     %15 = llvm.load %14 : !llvm<"i64*">
# CHECK-NEXT:     %16 = llvm.insertvalue %15, %13[3, 0] : !llvm<"{ i64*, i64*, i64, [1 x i64], [1 x i64] }">
# CHECK-NEXT:     %17 = llvm.mlir.constant({{[0-9]+}} : i64) : !llvm.i64
# CHECK-NEXT:     %18 = llvm.insertvalue %17, %16[4, 0] : !llvm<"{ i64*, i64*, i64, [1 x i64], [1 x i64] }">
# CHECK-NEXT:     %19 = llvm.extractvalue %18[1] : !llvm<"{ i64*, i64*, i64, [1 x i64], [1 x i64] }">
# CHECK-NEXT:     %20 = llvm.mlir.constant(0 : index) : !llvm.i64
# CHECK-NEXT:     %21 = llvm.mul %1, %0 : !llvm.i64
# CHECK-NEXT:     %22 = llvm.add %20, %21 : !llvm.i64
# CHECK-NEXT:     %23 = llvm.getelementptr %19[%22] : (!llvm<"i64*">, !llvm.i64) -> !llvm<"i64*">
# CHECK-NEXT:     %24 = llvm.load %23 : !llvm<"i64*">
# CHECK-NEXT:     llvm.return %24 : !llvm.i64
# CHECK-NEXT:   }
# CHECK-NEXT: }
