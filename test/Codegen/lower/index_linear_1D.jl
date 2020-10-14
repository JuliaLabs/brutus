# RUN: julia -e "import Brutus; Brutus.lit(:emit_lowered)" --startup-file=no %s 2>&1 | FileCheck %s

index(A, i) = A[i]
emit(index, Array{Int64, 1}, Int64)
#



# CHECK: module {
# CHECK-NEXT:   func @"Tuple{typeof(Main.index), Array{Int64, 1}, Int64}"(%arg0: !jlir<"typeof(Main.index)">, %arg1: memref<?xi64>, %arg2: i64) -> i64 {
# CHECK-NEXT:     %c1 = constant 1 : index
# CHECK-NEXT:     %0 = "jlir.convertstd"(%arg2) : (i64) -> index
# CHECK-NEXT:     %1 = subi %0, %c1 : index
# CHECK-NEXT:     %2 = load %arg1[%1] : memref<?xi64>
# CHECK-NEXT:     return %2 : i64
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK: module {
# CHECK-NEXT:   llvm.func @"Tuple{typeof(Main.index), Array{Int64, 1}, Int64}"(%arg0: !llvm<"%jl_value_t*">, %arg1: !llvm<"{ i64*, i64*, i64, [1 x i64], [1 x i64] }">, %arg2: !llvm.i64) -> !llvm.i64 {
# CHECK-NEXT:     %0 = llvm.mlir.constant(1 : index) : !llvm.i64
# CHECK-NEXT:     %1 = llvm.sub %arg2, %0 : !llvm.i64
# CHECK-NEXT:     %2 = llvm.extractvalue %arg1[1] : !llvm<"{ i64*, i64*, i64, [1 x i64], [1 x i64] }">
# CHECK-NEXT:     %3 = llvm.mlir.constant(0 : index) : !llvm.i64
# CHECK-NEXT:     %4 = llvm.mul %1, %0 : !llvm.i64
# CHECK-NEXT:     %5 = llvm.add %3, %4 : !llvm.i64
# CHECK-NEXT:     %6 = llvm.getelementptr %2[%5] : (!llvm<"i64*">, !llvm.i64) -> !llvm<"i64*">
# CHECK-NEXT:     %7 = llvm.load %6 : !llvm<"i64*">
# CHECK-NEXT:     llvm.return %7 : !llvm.i64
# CHECK-NEXT:   }
# CHECK-NEXT: }
