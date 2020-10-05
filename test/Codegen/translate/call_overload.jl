# RUN: julia -e "import Brutus; Brutus.lit(:emit_translated)" --startup-file=no %s 2>&1 | FileCheck %s



struct A
    x
end
(a::A)(y) = a.x + y
a = A(10)
emit(a, Int64)
# CHECK: func @"Tuple{Main.A, Int64}"(%arg0: !jlir.Main.A, %arg1: !jlir.Int64) -> !jlir.Any
# CHECK:   "jlir.goto"()[^bb1] : () -> ()
# CHECK: ^bb1:
# CHECK:   %0 = "jlir.constant"() {value = #jlir<"typeof(getfield)()">} : () -> !jlir<"typeof(getfield)">
# CHECK:   %1 = "jlir.constant"() {value = #jlir<":x">} : () -> !jlir.Symbol
# CHECK:   %2 = "jlir.call"(%0, %arg0, %1) : (!jlir<"typeof(getfield)">, !jlir.Main.A, !jlir.Symbol) -> !jlir.Any
# CHECK:   %3 = "jlir.constant"() {value = #jlir<"typeof(Base.:(+))()">} : () -> !jlir<"typeof(Base.:(+))">
# CHECK:   %4 = "jlir.call"(%3, %2, %arg1) : (!jlir<"typeof(Base.:(+))">, !jlir.Any, !jlir.Int64) -> !jlir.Any
# CHECK:   "jlir.return"(%4) : (!jlir.Any) -> ()
