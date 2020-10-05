# RUN: julia -e "import Brutus; Brutus.lit(:emit_optimized)" --startup-file=no %s 2>&1 | FileCheck %s

f() = return 2
emit(f)
# CHECK: func @"Tuple{typeof(Main.f)}"(%arg0: !jlir<"typeof(Main.f)">) -> !jlir.Int64
# CHECK:   "jlir.goto"()[^bb1] : () -> ()
# CHECK: ^bb1:
# CHECK:   %0 = "jlir.constant"() {value = #jlir<"2">} : () -> !jlir.Int64
# CHECK:   "jlir.return"(%0) : (!jlir.Int64) -> ()