# RUN: julia -e "import Brutus; Brutus.lit(:emit_lowered)" --startup-file=no %s 2>&1 | FileCheck %s

ne(x, y) = x != y
emit(ne, Float64, Float64)
# CHECK: func @"Tuple{typeof(Main.ne), Float64, Float64}"(%arg0: !jlir<"typeof(Main.ne)">, %arg1: f64, %arg2: f64) -> i1
# CHECK:   %0 = cmpf "une", %arg1, %arg2 : f64
# CHECK:   return %0 : i1
# CHECK: llvm.func @"Tuple{typeof(Main.ne), Float64, Float64}"(%arg0: !llvm<"%jl_value_t*">, %arg1: !llvm.double, %arg2: !llvm.double) -> !llvm.i1
# CHECK:   %0 = llvm.fcmp "une" %arg1, %arg2 : !llvm.double
# CHECK:   llvm.return %0 : !llvm.i1
