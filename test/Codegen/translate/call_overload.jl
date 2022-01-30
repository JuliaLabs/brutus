# RUN: julia -e "import Brutus; Brutus.lit(:emit_translated)" --startup-file=no %s 2>&1 | FileCheck %s

struct A
    x
end
(a::A)(y) = a.x + y
a = A(10)
emit(a, Int64)


# CHECK: Core.MethodMatch(Tuple{Main.Main.A, Int64}, svec(), (a::Main.Main.A)(y) in Main.Main at /{{.*}}/test/Codegen/translate/call_overload.jl:6, true)after translating to MLIR in JLIR dialect:module  {
# CHECK-NEXT:   func nested @"Tuple{Main.A, Int64}"(%arg0: !jlir.Main.A, %arg1: !jlir.Int64) -> !jlir.Any attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     "jlir.goto"()[^bb1] : () -> ()
# CHECK-NEXT:   ^bb1:  // pred: ^bb0
# CHECK-NEXT:     %0 = "jlir.constant"() {value = #jlir.getfield} : () -> !jlir<"typeof(getfield)">
# CHECK-NEXT:     %1 = "jlir.constant"() {value = #jlir<":x">} : () -> !jlir.Symbol
# CHECK-NEXT:     %2 = "jlir.call"(%0, %arg0, %1) : (!jlir<"typeof(getfield)">, !jlir.Main.A, !jlir.Symbol) -> !jlir.Any
# CHECK-NEXT:     %3 = "jlir.constant"() {value = #jlir<"Base.:(+)">} : () -> !jlir<"typeof(Base.:(+))">
# CHECK-NEXT:     %4 = "jlir.call"(%3, %2, %arg1) : (!jlir<"typeof(Base.:(+))">, !jlir.Any, !jlir.Int64) -> !jlir.Any
# CHECK-NEXT:     "jlir.return"(%4) : (!jlir.Any) -> ()
# CHECK-NEXT:   }
# CHECK-NEXT: }
