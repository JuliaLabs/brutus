# RUN: julia -e "import Brutus; Brutus.lit(:emit_optimized)" --startup-file=no %s 2>&1 | FileCheck %s

struct A
    x
end
(a::A)(y) = a.x + y
a = A(10)
emit(a, Int64)


# CHECK: module {
# CHECK-NEXT:   func @"Tuple{Main.A, Int64}"(%arg0: !jlir.Main.A, %arg1: !jlir.Int64) -> !jlir.Any {
# CHECK-NEXT:     "jlir.goto"()[^bb1] : () -> ()
# CHECK-NEXT:   ^bb1:  // pred: ^bb0
# CHECK-NEXT:     %0 = "jlir.constant"() {value = #jlir<":x">} : () -> !jlir.Symbol
# CHECK-NEXT:     %1 = "jlir.getfield"(%arg0, %0) : (!jlir.Main.A, !jlir.Symbol) -> !jlir.Any
# CHECK-NEXT:     %2 = "jlir.constant"() {value = #jlir<"typeof(Base.:(+))()">} : () -> !jlir<"typeof(Base.:(+))">
# CHECK-NEXT:     %3 = "jlir.call"(%2, %1, %arg1) : (!jlir<"typeof(Base.:(+))">, !jlir.Any, !jlir.Int64) -> !jlir.Any
# CHECK-NEXT:     "jlir.return"(%3) : (!jlir.Any) -> ()
# CHECK-NEXT:   }
# CHECK-NEXT: }
