# RUN: julia -e "import Brutus; Brutus.lit(:emit_translated)" --startup-file=no %s 2>&1 | FileCheck %s

# has the terminator unreachable
hasunreachable(x::Float64) = sqrt(x)
emit(hasunreachable, Float64)


# CHECK: module {
# CHECK-NEXT:   func @"Tuple{typeof(Main.hasunreachable), Float64}"(%arg0: !jlir<"typeof(Main.hasunreachable)">, %arg1: !jlir.Float64) -> !jlir.Float64 {
# CHECK-NEXT:     "jlir.goto"()[^bb1] : () -> ()
# CHECK-NEXT:   ^bb1:  // pred: ^bb0
# CHECK-NEXT:     %0 = "jlir.constant"() {value = #jlir<"#<intrinsic #33 lt_float>">} : () -> !jlir.Core.IntrinsicFunction
# CHECK-NEXT:     %1 = "jlir.constant"() {value = #jlir<"0">} : () -> !jlir.Float64
# CHECK-NEXT:     %2 = "jlir.call"(%0, %arg1, %1) : (!jlir.Core.IntrinsicFunction, !jlir.Float64, !jlir.Float64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%2)[^bb3, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb2:  // pred: ^bb1
# CHECK-NEXT:     %3 = "jlir.constant"() {value = #jlir<"typeof(Base.Math.throw_complex_domainerror)()">} : () -> !jlir<"typeof(Base.Math.throw_complex_domainerror)">
# CHECK-NEXT:     %4 = "jlir.constant"() {value = #jlir<":sqrt">} : () -> !jlir.Symbol
# CHECK-NEXT:     %5 = "jlir.invoke"(%3, %4, %arg1) {methodInstance = #jlir<"throw_complex_domainerror(Symbol, Float64) from throw_complex_domainerror(Symbol, Any)">} : (!jlir<"typeof(Base.Math.throw_complex_domainerror)">, !jlir.Symbol, !jlir.Float64) -> !jlir<"Union{}">
# CHECK-NEXT:     %6 = "jlir.undef"() : () -> !jlir.Float64
# CHECK-NEXT:     "jlir.return"(%6) : (!jlir.Float64) -> ()
# CHECK-NEXT:   ^bb3:  // pred: ^bb1
# CHECK-NEXT:     %7 = "jlir.constant"() {value = #jlir<"#<intrinsic #78 sqrt_llvm>">} : () -> !jlir.Core.IntrinsicFunction
# CHECK-NEXT:     %8 = "jlir.call"(%7, %arg1) : (!jlir.Core.IntrinsicFunction, !jlir.Float64) -> !jlir.Float64
# CHECK-NEXT:     "jlir.goto"()[^bb4] : () -> ()
# CHECK-NEXT:   ^bb4:  // pred: ^bb3
# CHECK-NEXT:     "jlir.return"(%8) : (!jlir.Float64) -> ()
# CHECK-NEXT:   }
# CHECK-NEXT: }
