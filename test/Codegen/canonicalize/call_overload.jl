# RUN: julia -e "import Brutus; Brutus.lit(:emit_optimized)" --startup-file=no %s 2>&1 | FileCheck %s

struct A
    x
end
(a::A)(y) = a.x + y
a = A(10)
emit(a, Int64)
# CHECK: func @"Tuple{Main.A, Int64}"(%arg0: !jlir.Main.A, %arg1: !jlir.Int64) -> !jlir.Any
# CHECK:   "jlir.goto"()[^bb1] : () -> ()
# CHECK: ^bb1:
# CHECK:   %0 = "jlir.constant"() {value = #jlir<":x">} : () -> !jlir.Symbol
# CHECK:   %1 = "jlir.getfield"(%arg0, %0) : (!jlir.Main.A, !jlir.Symbol) -> !jlir.Any
# CHECK:   %2 = "jlir.constant"() {value = #jlir<"typeof(Base.:(+))()">} : () -> !jlir<"typeof(Base.:(+))">
# CHECK:   %3 = "jlir.call"(%2, %1, %arg1) : (!jlir<"typeof(Base.:(+))">, !jlir.Any, !jlir.Int64) -> !jlir.Any
# CHECK:   "jlir.return"(%3) : (!jlir.Any) -> ()
