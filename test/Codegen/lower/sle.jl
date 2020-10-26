# RUN: julia -e "import Brutus; Brutus.lit(:emit_lowered)" --startup-file=no %s 2>&1 | FileCheck %s

sle_int(x, y) = Base.sle_int(x, y)
emit(sle_int, Int64, Int64)



# CHECK: module {
# CHECK-NEXT:   func @"Tuple{typeof(Main.sle_int), Int64, Int64}"(%arg0: !jlir<"typeof(Main.sle_int)">, %arg1: i64, %arg2: i64) -> i1 attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     %0 = cmpi "sle", %arg1, %arg2 : i64
# CHECK-NEXT:     return %0 : i1
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK: module {
# CHECK-NEXT:   llvm.func @"Tuple{typeof(Main.sle_int), Int64, Int64}"(%arg0: !llvm.ptr<struct<()>>, %arg1: !llvm.i64, %arg2: !llvm.i64) -> !llvm.i1 attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     %0 = llvm.icmp "sle" %arg1, %arg2 : !llvm.i64
# CHECK-NEXT:     llvm.return %0 : !llvm.i1
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @"_mlir_ciface_Tuple{typeof(Main.sle_int), Int64, Int64}"(%arg0: !llvm.ptr<struct<()>>, %arg1: !llvm.i64, %arg2: !llvm.i64) -> !llvm.i1 attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     %0 = llvm.call @"Tuple{typeof(Main.sle_int), Int64, Int64}"(%arg0, %arg1, %arg2) : (!llvm.ptr<struct<()>>, !llvm.i64, !llvm.i64) -> !llvm.i1
# CHECK-NEXT:     llvm.return %0 : !llvm.i1
# CHECK-NEXT:   }
# CHECK-NEXT: }
