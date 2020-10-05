# RUN: julia -e "import Brutus; Brutus.lit(:emit_optimized)" --startup-file=no %s 2>&1 | FileCheck %s

f(x) = x
emit(f, Int64)
# CHECK: func @"Tuple{typeof(Main.f), Int64}"(%arg0: !jlir<"typeof(Main.f)">, %arg1: !jlir.Int64) -> !jlir.Int64
# CHECK:   "jlir.goto"()[^bb1] : () -> ()
# CHECK: ^bb1:
# CHECK:   "jlir.return"(%arg1) : (!jlir.Int64) -> ()
