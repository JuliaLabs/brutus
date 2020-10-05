# RUN: julia -e "import Brutus; Brutus.lit(:emit_optimized)" --startup-file=no %s 2>&1 | FileCheck %s

function haspi(x::Union{Int64, Float64})
    if x isa Int64
        return x + 1
    end
end
emit(haspi, Union{Int64, Float64})
# CHECK: func @"Tuple{typeof(Main.haspi), Union{Float64, Int64}}"(%arg0: !jlir<"typeof(Main.haspi)">, %arg1: !jlir<"Union{Float64, Int64}">) -> !jlir<"Union{Nothing, Int64}">
# CHECK:   "jlir.goto"()[^bb1] : () -> ()
# CHECK: ^bb1:
# CHECK:   %0 = "jlir.constant"() {value = #jlir.Int64} : () -> !jlir.DataType
# CHECK:   %1 = "jlir.isa"(%arg1, %0) : (!jlir<"Union{Float64, Int64}">, !jlir.DataType) -> !jlir.Bool
# CHECK:   "jlir.gotoifnot"(%1)[^bb3, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK: ^bb2:
# CHECK:   %2 = "jlir.pi"(%arg1) : (!jlir<"Union{Float64, Int64}">) -> !jlir.Int64
# CHECK:   %3 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK:   %4 = "jlir.add_int"(%2, %3) : (!jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK:   %5 = "jlir.pi"(%4) : (!jlir.Int64) -> !jlir<"Union{Nothing, Int64}">
# CHECK:   "jlir.return"(%5) : (!jlir<"Union{Nothing, Int64}">) -> ()
# CHECK: ^bb3:
# CHECK:   %6 = "jlir.constant"() {value = #jlir.nothing} : () -> !jlir.Nothing
# CHECK:   %7 = "jlir.pi"(%6) : (!jlir.Nothing) -> !jlir<"Union{Nothing, Int64}">
# CHECK:   "jlir.return"(%7) : (!jlir<"Union{Nothing, Int64}">) -> ()
