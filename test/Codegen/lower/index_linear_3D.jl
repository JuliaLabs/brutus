# RUN: julia -e "import Brutus; Brutus.lit(:emit_lowered)" --startup-file=no %s 2>&1 | FileCheck %s

index(A, i) = A[i]
emit(index, Array{Int64, 3}, Int64)
#



# CHECK: module {
# CHECK-NEXT:   func @"Tuple{typeof(Main.index), Array{Int64, 3}, Int64}"(%arg0: !jlir<"typeof(Main.index)">, %arg1: memref<?x?x?xi64>, %arg2: i64) -> i64 {
# CHECK-NEXT:     %c0 = constant 0 : index
# CHECK-NEXT:     %c1 = constant 1 : index
# CHECK-NEXT:     %0 = "jlir.convertstd"(%arg2) : (i64) -> index
# CHECK-NEXT:     %1 = subi %0, %c1 : index
# CHECK-NEXT:     %2 = load %arg1[%c0, %c0, %1] : memref<?x?x?xi64>
# CHECK-NEXT:     return %2 : i64
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK: module {
# CHECK-NEXT:   llvm.func @"Tuple{typeof(Main.index), Array{Int64, 3}, Int64}"(%arg0: !llvm<"%jl_value_t*">, %arg1: !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">, %arg2: !llvm.i64) -> !llvm.i64 {
# CHECK-NEXT:     %0 = llvm.mlir.constant(0 : index) : !llvm.i64
# CHECK-NEXT:     %1 = llvm.mlir.constant(1 : index) : !llvm.i64
# CHECK-NEXT:     %2 = llvm.sub %arg2, %1 : !llvm.i64
# CHECK-NEXT:     %3 = llvm.extractvalue %arg1[1] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %4 = llvm.extractvalue %arg1[4, 0] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %5 = llvm.mul %0, %4 : !llvm.i64
# CHECK-NEXT:     %6 = llvm.add %0, %5 : !llvm.i64
# CHECK-NEXT:     %7 = llvm.extractvalue %arg1[4, 1] : !llvm<"{ i64*, i64*, i64, [3 x i64], [3 x i64] }">
# CHECK-NEXT:     %8 = llvm.mul %0, %7 : !llvm.i64
# CHECK-NEXT:     %9 = llvm.add %6, %8 : !llvm.i64
# CHECK-NEXT:     %10 = llvm.mul %2, %1 : !llvm.i64
# CHECK-NEXT:     %11 = llvm.add %9, %10 : !llvm.i64
# CHECK-NEXT:     %12 = llvm.getelementptr %3[%11] : (!llvm<"i64*">, !llvm.i64) -> !llvm<"i64*">
# CHECK-NEXT:     %13 = llvm.load %12 : !llvm<"i64*">
# CHECK-NEXT:     llvm.return %13 : !llvm.i64
# CHECK-NEXT:   }
# CHECK-NEXT: }
