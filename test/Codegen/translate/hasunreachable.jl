# RUN: julia -e "import Brutus; Brutus.lit(:emit_translated)" --startup-file=no %s 2>&1 | FileCheck %s



# has the terminator unreachable
hasunreachable(x::Float64) = sqrt(x)
emit(hasunreachable, Float64)
# CHECK: func @"Tuple{typeof(Main.hasunreachable), Float64}"(%arg0: !jlir<"typeof(Main.hasunreachable)">, %arg1: !jlir.Float64) -> !jlir.Float64 {
# CHECK:   "jlir.goto"()[^bb1] : () -> ()
# CHECK: ^bb1:  // pred: ^bb0
# CHECK:   %0 = "jlir.constant"() {value = #jlir<"#<intrinsic #33 lt_float>">} : () -> !jlir.Core.IntrinsicFunction
# CHECK:   %1 = "jlir.constant"() {value = #jlir<"0">} : () -> !jlir.Float64
# CHECK:   %2 = "jlir.call"(%0, %arg1, %1) : (!jlir.Core.IntrinsicFunction, !jlir.Float64, !jlir.Float64) -> !jlir.Bool
# CHECK:   "jlir.gotoifnot"(%2)[^bb3, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK: ^bb2:  // pred: ^bb1
# CHECK:   %3 = "jlir.constant"() {value = #jlir<"typeof(Base.Math.throw_complex_domainerror)()">} : () -> !jlir<"typeof(Base.Math.throw_complex_domainerror)">
# CHECK:   %4 = "jlir.constant"() {value = #jlir<":sqrt">} : () -> !jlir.Symbol
# CHECK:   %5 = "jlir.invoke"(%3, %4, %arg1) {methodInstance = #jlir<"throw_complex_domainerror(Symbol, Float64)">} : (!jlir<"typeof(Base.Math.throw_complex_domainerror)">, !jlir.Symbol, !jlir.Float64) -> !jlir<"Union{}">
# CHECK:   %6 = "jlir.undef"() : () -> !jlir.Float64
# CHECK:   "jlir.return"(%6) : (!jlir.Float64) -> ()
# CHECK: ^bb3:  // pred: ^bb1
# CHECK:   %7 = "jlir.constant"() {value = #jlir<"#<intrinsic #78 sqrt_llvm>">} : () -> !jlir.Core.IntrinsicFunction
# CHECK:   %8 = "jlir.call"(%7, %arg1) : (!jlir.Core.IntrinsicFunction, !jlir.Float64) -> !jlir.Float64
# CHECK:   "jlir.goto"()[^bb4] : () -> ()
# CHECK: ^bb4:  // pred: ^bb3
# CHECK:   "jlir.return"(%8) : (!jlir.Float64) -> ()
