# RUN: julia -e "import Brutus; Brutus.lit(:emit_translated)" --startup-file=no %s 2>&1 | FileCheck %s





function haspi(x::Union{Int64, Float64})
    if x isa Int64
        return x + 1
    end
end
emit(haspi, Union{Int64, Float64})
# CHECK: func @"Tuple{typeof(Main.haspi), Union{Float64, Int64}}"(%arg0: !jlir<"typeof(Main.haspi)">, %arg1: !jlir<"Union{Float64, Int64}">) -> !jlir<"Union{Nothing, Int64}">
# CHECK:   "jlir.goto"()[^bb1] : () -> ()
# CHECK: ^bb1:
# CHECK:   %0 = "jlir.constant"() {value = #jlir<"typeof(isa)()">} : () -> !jlir<"typeof(isa)">
# CHECK:   %1 = "jlir.constant"() {value = #jlir.Int64} : () -> !jlir.DataType
# CHECK:   %2 = "jlir.call"(%0, %arg1, %1) : (!jlir<"typeof(isa)">, !jlir<"Union{Float64, Int64}">
# CHECK:   "jlir.gotoifnot"(%2)[^bb3, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK: ^bb2:
# CHECK:   %3 = "jlir.pi"(%arg1) : (!jlir<"Union{Float64, Int64}">) -> !jlir.Int64
# CHECK:   %4 = "jlir.constant"() {value = #jlir<"#<intrinsic #2 add_int>">} : () -> !jlir.Core.IntrinsicFunction
# CHECK:   %5 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK:   %6 = "jlir.call"(%4, %3, %5) : (!jlir.Core.IntrinsicFunction, !jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK:   %7 = "jlir.pi"(%6) : (!jlir.Int64) -> !jlir<"Union{Nothing, Int64}">
# CHECK:   "jlir.return"(%7) : (!jlir<"Union{Nothing, Int64}">) -> ()
# CHECK: ^bb3:
# CHECK:   %8 = "jlir.constant"() {value = #jlir.nothing} : () -> !jlir.Nothing
# CHECK:   %9 = "jlir.pi"(%8) : (!jlir.Nothing) -> !jlir<"Union{Nothing, Int64}">
# CHECK:   "jlir.return"(%9) : (!jlir<"Union{Nothing, Int64}">) -> ()
