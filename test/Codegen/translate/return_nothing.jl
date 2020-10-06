# RUN: julia -e "import Brutus; Brutus.lit(:emit_translated)" --startup-file=no %s 2>&1 | FileCheck %s

f() = return
emit(f)


# CHECK: module {
# CHECK-NEXT:   func @"Tuple{typeof(Main.f)}"(%arg0: !jlir<"typeof(Main.f)">) -> !jlir.Nothing {
# CHECK-NEXT:     "jlir.goto"()[^bb1] : () -> ()
# CHECK-NEXT:   ^bb1:  // pred: ^bb0
# CHECK-NEXT:     %0 = "jlir.constant"() {value = #jlir.nothing} : () -> !jlir.Nothing
# CHECK-NEXT:     "jlir.return"(%0) : (!jlir.Nothing) -> ()
# CHECK-NEXT:   }
# CHECK-NEXT: }
