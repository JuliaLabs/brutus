# RUN: julia -e "import Brutus; Brutus.lit(:emit_lowered)" --startup-file=no %s 2>&1 | FileCheck %s

index(A, i) = A[i]
emit(index, Array{Int64, 1}, Int64)
#




# CHECK: module  {
# CHECK-NEXT:   func nested @"Tuple{typeof(Main.index), Array{Int64, 1}, Int64}"(%arg0: !jlir<"typeof(Main.index)">, %arg1: !jlir<"Array{Int64, 1}">, %arg2: !jlir.Int64) -> !jlir.Int64 attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     "jlir.goto"()[^bb1] : () -> ()
# CHECK-NEXT:   ^bb1:  // pred: ^bb0
# CHECK-NEXT:     %0 = "jlir.constant"() {value = #jlir.Core.arrayref} : () -> !jlir<"typeof(Core.arrayref)">
# CHECK-NEXT:     %1 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     %2 = "jlir.call"(%0, %1, %arg1, %arg2) : (!jlir<"typeof(Core.arrayref)">, !jlir.Bool, !jlir<"Array{Int64, 1}">, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     "jlir.return"(%2) : (!jlir.Int64) -> ()
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK: module  {
# CHECK-NEXT:   func nested @"Tuple{typeof(Main.index), Array{Int64, 1}, Int64}"(%arg0: !jlir<"typeof(Main.index)">, %arg1: memref<?xi64>, %arg2: i64) -> i64 attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     %true = constant true
# CHECK-NEXT:     %0 = "jlir.convertstd"(%arg1) : (memref<?xi64>) -> !jlir<"Array{Int64, 1}">
# CHECK-NEXT:     %1 = "jlir.convertstd"(%arg2) : (i64) -> !jlir.Int64
# CHECK-NEXT:     %2 = "jlir.convertstd"(%true) : (i1) -> !jlir.Bool
# CHECK-NEXT:     %3 = "jlir.arrayref"(%2, %0, %1) : (!jlir.Bool, !jlir<"Array{Int64, 1}">, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %4 = "jlir.convertstd"(%3) : (!jlir.Int64) -> i64
# CHECK-NEXT:     return %4 : i64
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK: error: lowering to LLVM dialect failed
# CHECK-NEXT: error: module verification failed

# CHECK: module  {
# CHECK-NEXT:   llvm.func @"Tuple{typeof(Main.index), Array{Int64, 1}, Int64}"(%arg0: !llvm.ptr<struct<"struct_jl_value_type", opaque>>, %arg1: !llvm.ptr<i64>, %arg2: !llvm.ptr<i64>, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64) -> i64 attributes {llvm.emit_c_interface, sym_visibility = "nested"} {
# CHECK-NEXT:     %0 = llvm.mlir.undef : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
# CHECK-NEXT:     %1 = llvm.insertvalue %arg1, %0[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
# CHECK-NEXT:     %2 = llvm.insertvalue %arg2, %1[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
# CHECK-NEXT:     %3 = llvm.insertvalue %arg3, %2[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
# CHECK-NEXT:     %4 = llvm.insertvalue %arg4, %3[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
# CHECK-NEXT:     %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
# CHECK-NEXT:     %6 = llvm.mlir.constant(true) : i1
# CHECK-NEXT:     %7 = "jlir.arrayref"(%6, %5, %arg6) : (i1, !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>, i64) -> !jlir.Int64
# CHECK-NEXT:     llvm.return %7 : !jlir.Int64
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @"_mlir_ciface_Tuple{typeof(Main.index), Array{Int64, 1}, Int64}"(%arg0: !llvm.ptr<struct<"struct_jl_value_type", opaque>>, %arg1: !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>, %arg2: i64) -> i64 attributes {llvm.emit_c_interface, sym_visibility = "nested"} {
# CHECK-NEXT:     %0 = llvm.load %arg1 : !llvm.ptr<struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>>
# CHECK-NEXT:     %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
# CHECK-NEXT:     %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
# CHECK-NEXT:     %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
# CHECK-NEXT:     %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
# CHECK-NEXT:     %5 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr<i64>, ptr<i64>, i64, array<1 x i64>, array<1 x i64>)>
# CHECK-NEXT:     %6 = llvm.call @"Tuple{typeof(Main.index), Array{Int64, 1}, Int64}"(%arg0, %1, %2, %3, %4, %5, %arg2) : (!llvm.ptr<struct<"struct_jl_value_type", opaque>>, !llvm.ptr<i64>, !llvm.ptr<i64>, i64, i64, i64, i64) -> i64
# CHECK-NEXT:     llvm.return %6 : i64
# CHECK-NEXT:   }
# CHECK-NEXT: }
