# RUN: julia -e "import Brutus; Brutus.lit(:emit_lowered)" --startup-file=no %s 2>&1 | FileCheck %s

index(A, i, j, k) = A[i, j, k]
emit(index, Array{Int64, 3}, Int64, Int64, Int64)
#



# CHECK: module {
# CHECK-NEXT:   func @"Tuple{typeof(Main.index), Array{Int64, 3}, Int64, Int64, Int64}"(%arg0: !jlir<"typeof(Main.index)">, %arg1: memref<?x?x?xi64>, %arg2: i64, %arg3: i64, %arg4: i64) -> i64 {
# CHECK-NEXT:     %c1 = constant 1 : index
# CHECK-NEXT:     %0 = "jlir.convertstd"(%arg2) : (i64) -> index
# CHECK-NEXT:     %1 = subi %0, %c1 : index
# CHECK-NEXT:     %2 = "jlir.convertstd"(%arg3) : (i64) -> index
# CHECK-NEXT:     %3 = subi %2, %c1 : index
# CHECK-NEXT:     %4 = "jlir.convertstd"(%arg4) : (i64) -> index
# CHECK-NEXT:     %5 = subi %4, %c1 : index
# CHECK-NEXT:     %6 = load %arg1[%5, %3, %1] : memref<?x?x?xi64>
# CHECK-NEXT:     return %6 : i64
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK: module {
# CHECK-NEXT:   llvm.func @"Tuple{typeof(Main.index), Array{Int64, 3}, Int64, Int64, Int64}"(%arg0: !llvm<"%jl_value_t*">, %arg1: !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">, %arg2: !llvm.i64, %arg3: !llvm.i64, %arg4: !llvm.i64) -> !llvm.i64 {
# CHECK-NEXT:     %0 = llvm.mlir.constant(1 : index) : !llvm.i64
# CHECK-NEXT:     %1 = llvm.sub %arg2, %0 : !llvm.i64
# CHECK-NEXT:     %2 = llvm.sub %arg3, %0 : !llvm.i64
# CHECK-NEXT:     %3 = llvm.sub %arg4, %0 : !llvm.i64
# CHECK-NEXT:     %4 = llvm.extractvalue %arg1[1] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %5 = llvm.mlir.constant(0 : index) : !llvm.i64
# CHECK-NEXT:     %6 = llvm.extractvalue %arg1[4, 0] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %7 = llvm.mul %3, %6 : !llvm.i64
# CHECK-NEXT:     %8 = llvm.add %5, %7 : !llvm.i64
# CHECK-NEXT:     %9 = llvm.extractvalue %arg1[4, 1] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %10 = llvm.mul %2, %9 : !llvm.i64
# CHECK-NEXT:     %11 = llvm.add %8, %10 : !llvm.i64
# CHECK-NEXT:     %12 = llvm.mul %1, %0 : !llvm.i64
# CHECK-NEXT:     %13 = llvm.add %11, %12 : !llvm.i64
# CHECK-NEXT:     %14 = llvm.getelementptr %4[%13] : (!llvm<"i64*">, !llvm.i64) -> !llvm<"i64*">
# CHECK-NEXT:     %15 = llvm.load %14 : !llvm<"i64*">
# CHECK-NEXT:     llvm.return %15 : !llvm.i64
# CHECK-NEXT:   }
# CHECK-NEXT: }
