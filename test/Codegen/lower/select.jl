# RUN: julia -e "import Brutus; Brutus.lit(:emit_lowered)" --startup-file=no %s 2>&1 | FileCheck %s

select(c) = 1 + (c ? 2 : 3)
emit(select, Bool)



# CHECK: module {
# CHECK-NEXT:   func @"Tuple{typeof(Main.select), Bool}"(%arg0: !jlir<"typeof(Main.select)">, %arg1: i1) -> i64 {
# CHECK-NEXT:     %c2_i64 = constant 2 : i64
# CHECK-NEXT:     %c3_i64 = constant 3 : i64
# CHECK-NEXT:     %c1_i64 = constant 1 : i64
# CHECK-NEXT:     %0 = select %arg1, %c2_i64, %c3_i64 : i64
# CHECK-NEXT:     %1 = addi %0, %c1_i64 : i64
# CHECK-NEXT:     return %1 : i64
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK: module {
# CHECK-NEXT:   llvm.func @"Tuple{typeof(Main.select), Bool}"(%arg0: !llvm.ptr<struct<"jl_value_t", ()>>, %arg1: !llvm.i1) -> !llvm.i64 {
# CHECK-NEXT:     %0 = llvm.mlir.constant({{[0-9]+}} : i64) : !llvm.i64
# CHECK-NEXT:     %1 = llvm.mlir.constant({{[0-9]+}} : i64) : !llvm.i64
# CHECK-NEXT:     %2 = llvm.mlir.constant({{[0-9]+}} : i64) : !llvm.i64
# CHECK-NEXT:     %3 = llvm.select %arg1, %0, %1 : !llvm.i1, !llvm.i64
# CHECK-NEXT:     %4 = llvm.add %3, %2 : !llvm.i64
# CHECK-NEXT:     llvm.return %4 : !llvm.i64
# CHECK-NEXT:   }
# CHECK-NEXT: }
