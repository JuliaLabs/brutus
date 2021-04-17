# RUN: julia -e "import Brutus; Brutus.lit(:emit_lowered)" --startup-file=no %s 2>&1 | FileCheck %s

index(A, i, j, k) = A[i, j, k]
emit(index, Array{Int64, 3}, Int64, Int64, Int64)
#



# CHECK:   func nested @"Tuple{typeof(Main.index), Array{Int64, 3}, Int64, Int64, Int64}"(%arg0: !jlir<"typeof(Main.index)">, %arg1: !jlir<"Array{Int64, 3}">, %arg2: !jlir.Int64, %arg3: !jlir.Int64, %arg4: !jlir.Int64) -> !jlir.Int64 attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     "jlir.goto"()[^bb1] : () -> ()
# CHECK-NEXT:   ^bb1:  // pred: ^bb0
# CHECK-NEXT:     %0 = "jlir.constant"() {value = #jlir.Core.arrayref} : () -> !jlir<"typeof(Core.arrayref)">
# CHECK-NEXT:     %1 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     %2 = "jlir.call"(%0, %1, %arg1, %arg2, %arg3, %arg4) : (!jlir<"typeof(Core.arrayref)">, !jlir.Bool, !jlir<"Array{Int64, 3}">, !jlir.Int64, !jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     "jlir.return"(%2) : (!jlir.Int64) -> ()
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK: module  {
# CHECK-NEXT:   func nested @"Tuple{typeof(Main.index), Array{Int64, 3}, Int64, Int64, Int64}"(%arg0: !jlir<"typeof(Main.index)">, %arg1: memref<?x?x?xi64>, %arg2: i64, %arg3: i64, %arg4: i64) -> i64 attributes {llvm.emit_c_interface} {
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

# CHECK:   llvm.func @"Tuple{typeof(Main.index), Array{Int64, 3}, Int64, Int64, Int64}"(%arg0: !llvm.ptr<struct<"struct_jl_value_type", opaque>>, %arg1: !llvm.ptr<i64>, %arg2: !llvm.ptr<i64>, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64) -> i64 attributes {llvm.emit_c_interface, sym_visibility = "nested"} {
# CHECK-NEXT:     %0 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<3 x i64>, array<3 x i64>)>
# CHECK-NEXT:     %1 = llvm.insertvalue %arg1, %0[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<3 x i64>, array<3 x i64>)>
# CHECK-NEXT:     %2 = llvm.insertvalue %arg2, %1[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<3 x i64>, array<3 x i64>)>
# CHECK-NEXT:     %3 = llvm.insertvalue %arg3, %2[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<3 x i64>, array<3 x i64>)>
# CHECK-NEXT:     %4 = llvm.insertvalue %arg4, %3[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<3 x i64>, array<3 x i64>)>
# CHECK-NEXT:     %5 = llvm.insertvalue %arg7, %4[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<3 x i64>, array<3 x i64>)>
# CHECK-NEXT:     %6 = llvm.insertvalue %arg5, %5[3, 1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<3 x i64>, array<3 x i64>)>
# CHECK-NEXT:     %7 = llvm.insertvalue %arg8, %6[4, 1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<3 x i64>, array<3 x i64>)>
# CHECK-NEXT:     %8 = llvm.insertvalue %arg6, %7[3, 2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<3 x i64>, array<3 x i64>)>
# CHECK-NEXT:     %9 = llvm.insertvalue %arg9, %8[4, 2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<3 x i64>, array<3 x i64>)>
# CHECK-NEXT:     %10 = llvm.mlir.constant(1 : index) : i64
# CHECK-NEXT:     %11 = llvm.sub %arg10, %10  : i64
# CHECK-NEXT:     %12 = llvm.sub %arg11, %10  : i64
# CHECK-NEXT:     %13 = llvm.sub %arg12, %10  : i64
# CHECK-NEXT:     %14 = llvm.extractvalue %9[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<3 x i64>, array<3 x i64>)>
# CHECK-NEXT:     %15 = llvm.extractvalue %9[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<3 x i64>, array<3 x i64>)>
# CHECK-NEXT:     %16 = llvm.mul %13, %15  : i64
# CHECK-NEXT:     %17 = llvm.extractvalue %9[4, 1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<3 x i64>, array<3 x i64>)>
# CHECK-NEXT:     %18 = llvm.mul %12, %17  : i64
# CHECK-NEXT:     %19 = llvm.add %16, %18  : i64
# CHECK-NEXT:     %20 = llvm.add %19, %11  : i64
# CHECK-NEXT:     %21 = llvm.getelementptr %14[%20] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
# CHECK-NEXT:     %22 = llvm.load %21 : !llvm.ptr<i64>
# CHECK-NEXT:     llvm.return %22 : i64
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @"_mlir_ciface_Tuple{typeof(Main.index), Array{Int64, 3}, Int64, Int64, Int64}"(%arg0: !llvm.ptr<struct<"struct_jl_value_type", opaque>>, %arg1: !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<3 x i64>, array<3 x i64>)>>, %arg2: i64, %arg3: i64, %arg4: i64) -> i64 attributes {llvm.emit_c_interface, sym_visibility = "nested"} {
# CHECK-NEXT:     %0 = llvm.load %arg1 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<3 x i64>, array<3 x i64>)>>
# CHECK-NEXT:     %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<3 x i64>, array<3 x i64>)>
# CHECK-NEXT:     %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<3 x i64>, array<3 x i64>)>
# CHECK-NEXT:     %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<3 x i64>, array<3 x i64>)>
# CHECK-NEXT:     %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<3 x i64>, array<3 x i64>)>
# CHECK-NEXT:     %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<3 x i64>, array<3 x i64>)>
# CHECK-NEXT:     %6 = llvm.extractvalue %0[3, 2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<3 x i64>, array<3 x i64>)>
# CHECK-NEXT:     %7 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<3 x i64>, array<3 x i64>)>
# CHECK-NEXT:     %8 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<3 x i64>, array<3 x i64>)>
# CHECK-NEXT:     %9 = llvm.extractvalue %0[4, 2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<3 x i64>, array<3 x i64>)>
# CHECK-NEXT:     %10 = llvm.call @"Tuple{typeof(Main.index), Array{Int64, 3}, Int64, Int64, Int64}"(%arg0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %arg2, %arg3, %arg4) : (!llvm.ptr<struct<"struct_jl_value_type", opaque>>, !llvm.ptr<i64>, !llvm.ptr<i64>, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> i64
# CHECK-NEXT:     llvm.return %10 : i64
# CHECK-NEXT:   }
# CHECK-NEXT: }
