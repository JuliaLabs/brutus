# RUN: julia -e "import Brutus; Brutus.lit(:emit_lowered)" --startup-file=no %s 2>&1 | FileCheck %s

add(x, y) = x + y
emit(add, Float64, Float64)
# CHECK: func @"Tuple{typeof(Main.add), Float64, Float64}"(%arg0: !jlir<"typeof(Main.add)">, %arg1: f64, %arg2: f64) -> f64
# CHECK:   %0 = addf %arg1, %arg2 : f64
# CHECK:   return %0 : f64
# CHECK: llvm.func @"Tuple{typeof(Main.add), Float64, Float64}"(%arg0: !llvm<"%jl_value_t*">, %arg1: !llvm.double, %arg2: !llvm.double) -> !llvm.double
# CHECK:   %0 = llvm.fadd %arg1, %arg2 : !llvm.double
# CHECK:   llvm.return %0 : !llvm.double
