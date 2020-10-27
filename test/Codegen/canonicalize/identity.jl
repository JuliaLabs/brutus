# RUN: julia -e "import Brutus; Brutus.lit(:emit_optimized)" --startup-file=no %s 2>&1 | FileCheck %s

f(x) = x
emit(f, Int64)


# CHECK: module {
# CHECK-NEXT:   func @"Tuple{typeof(Main.f), Int64}"(%arg0: !jlir<"typeof(Main.f)">, %arg1: !jlir.Int64) -> !jlir.Int64 attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     "jlir.goto"()[^bb1] : () -> ()
# CHECK-NEXT:   ^bb1:  // pred: ^bb0
# CHECK-NEXT:     "jlir.return"(%arg1) : (!jlir.Int64) -> ()
# CHECK-NEXT:   }
# CHECK-NEXT: }
