# RUN: julia -e "import Brutus; Brutus.lit(:emit_lowered)" --startup-file=no %s 2>&1 | FileCheck %s

index(A, i) = A[i]
emit(index, Array{Int64, 1}, Int64)
#



# CHECK: module {
# CHECK-NEXT:   func @"Tuple{typeof(Main.index), Array{Int64, 1}, Int64}"(%arg0: !jlir<"typeof(Main.index)">, %arg1: memref<?xi64>, %arg2: i64) -> i64 attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     %c1 = constant 1 : index
# CHECK-NEXT:     %0 = "jlir.convertstd"(%arg2) : (i64) -> index
# CHECK-NEXT:     %1 = subi %0, %c1 : index
# CHECK-NEXT:     %2 = load %arg1[%1] : memref<?xi64>
# CHECK-NEXT:     return %2 : i64
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK: module {
# CHECK-NEXT:   llvm.func @"Tuple{typeof(Main.index), Array{Int64, 1}, Int64}"(%arg0: !llvm.ptr<struct<()>>, %arg1: !llvm.ptr<i64>, %arg2: !llvm.ptr<i64>, %arg3: !llvm.i64, %arg4: !llvm.i64, %arg5: !llvm.i64, %arg6: !llvm.i64) -> !llvm.i64 attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     %0 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
# CHECK-NEXT:     %1 = llvm.insertvalue %arg1, %0[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
# CHECK-NEXT:     %2 = llvm.insertvalue %arg2, %1[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
# CHECK-NEXT:     %3 = llvm.insertvalue %arg3, %2[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
# CHECK-NEXT:     %4 = llvm.insertvalue %arg4, %3[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
# CHECK-NEXT:     %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
# CHECK-NEXT:     %6 = llvm.mlir.constant(1 : index) : !llvm.i64
# CHECK-NEXT:     %7 = llvm.sub %arg6, %6 : !llvm.i64
# CHECK-NEXT:     %8 = llvm.extractvalue %5[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
# CHECK-NEXT:     %9 = llvm.mlir.constant(0 : index) : !llvm.i64
# CHECK-NEXT:     %10 = llvm.mul %7, %6 : !llvm.i64
# CHECK-NEXT:     %11 = llvm.add %9, %10 : !llvm.i64
# CHECK-NEXT:     %12 = llvm.getelementptr %8[%11] : (!llvm.ptr<i64>, !llvm.i64) -> !llvm.ptr<i64>
# CHECK-NEXT:     %13 = llvm.load %12 : !llvm.ptr<i64>
# CHECK-NEXT:     llvm.return %13 : !llvm.i64
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @"_mlir_ciface_Tuple{typeof(Main.index), Array{Int64, 1}, Int64}"(%arg0: !llvm.ptr<struct<()>>, %arg1: !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>, %arg2: !llvm.i64) -> !llvm.i64 attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     %0 = llvm.load %arg1 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
# CHECK-NEXT:     %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
# CHECK-NEXT:     %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
# CHECK-NEXT:     %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
# CHECK-NEXT:     %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
# CHECK-NEXT:     %5 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
# CHECK-NEXT:     %6 = llvm.call @"Tuple{typeof(Main.index), Array{Int64, 1}, Int64}"(%arg0, %1, %2, %3, %4, %5, %arg2) : (!llvm.ptr<struct<()>>, !llvm.ptr<i64>, !llvm.ptr<i64>, !llvm.i64, !llvm.i64, !llvm.i64, !llvm.i64) -> !llvm.i64
# CHECK-NEXT:     llvm.return %6 : !llvm.i64
# CHECK-NEXT:   }
# CHECK-NEXT: }
