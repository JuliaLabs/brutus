# RUN: julia -e "import Brutus; Brutus.lit(:emit_optimized)" --startup-file=no %s 2>&1 | FileCheck %s

f() = return
emit(f)
# CHECK: func @"Tuple{typeof(Main.f)}"(%arg0: !jlir<"typeof(Main.f)">) -> !jlir.Nothing
# CHECK:   "jlir.goto"()[^bb1] : () -> ()
# CHECK: ^bb1:
# CHECK:   %0 = "jlir.constant"() {value = #jlir.nothing} : () -> !jlir.Nothing
# CHECK:   "jlir.return"(%0) : (!jlir.Nothing) -> ()
