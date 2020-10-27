# RUN: julia -e "import Brutus; Brutus.lit(:emit_lowered)" --startup-file=no %s 2>&1 | FileCheck %s

add(x, y) = x + y
emit(add, Int64, Int64)



# CHECK: module {
# CHECK-NEXT:   func @"Tuple{typeof(Main.add), Int64, Int64}"(%arg0: !jlir<"typeof(Main.add)">, %arg1: i64, %arg2: i64) -> i64 attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     %0 = addi %arg1, %arg2 : i64
# CHECK-NEXT:     return %0 : i64
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK: module {
# CHECK-NEXT:   llvm.func @"Tuple{typeof(Main.add), Int64, Int64}"(%arg0: !llvm.ptr<struct<()>>, %arg1: !llvm.i64, %arg2: !llvm.i64) -> !llvm.i64 attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     %0 = llvm.add %arg1, %arg2 : !llvm.i64
# CHECK-NEXT:     llvm.return %0 : !llvm.i64
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @"_mlir_ciface_Tuple{typeof(Main.add), Int64, Int64}"(%arg0: !llvm.ptr<struct<()>>, %arg1: !llvm.i64, %arg2: !llvm.i64) -> !llvm.i64 attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     %0 = llvm.call @"Tuple{typeof(Main.add), Int64, Int64}"(%arg0, %arg1, %arg2) : (!llvm.ptr<struct<()>>, !llvm.i64, !llvm.i64) -> !llvm.i64
# CHECK-NEXT:     llvm.return %0 : !llvm.i64
# CHECK-NEXT:   }
# CHECK-NEXT: }
