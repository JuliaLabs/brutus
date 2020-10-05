# RUN: julia -e "import Brutus; Brutus.lit(:emit_lowered)" --startup-file=no %s 2>&1 | FileCheck %s

add(x, y) = x + y
emit(add, Int64, Int64)
# CHECK: func @"Tuple{typeof(Main.add), Int64, Int64}"(%arg0: !jlir<"typeof(Main.add)">, %arg1: i64, %arg2: i64) -> i64
# CHECK:   %0 = addi %arg1, %arg2 : i64
# CHECK:   return %0 : i64
# CHECK: llvm.func @"Tuple{typeof(Main.add), Int64, Int64}"(%arg0: !llvm<"%jl_value_t*">, %arg1: !llvm.i64, %arg2: !llvm.i64) -> !llvm.i64
# CHECK:   %0 = llvm.add %arg1, %arg2 : !llvm.i64
# CHECK:   llvm.return %0 : !llvm.i64