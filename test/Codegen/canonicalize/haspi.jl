# RUN: julia -e "import Brutus; Brutus.lit(:emit_optimized)" --startup-file=no %s 2>&1 | FileCheck %s

function haspi(x::Union{Int64, Float64})
    if x isa Int64
        return x + 1
    end
end
emit(haspi, Union{Int64, Float64})


# CHECK:   func nested @"Tuple{typeof(Main.haspi), Union{Float64, Int64}}"(%arg0: !jlir<"typeof(Main.haspi)">, %arg1: !jlir<"Union{Float64, Int64}">) -> !jlir<"Union{Nothing, Int64}"> attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     "jlir.goto"()[^bb1] : () -> ()
# CHECK-NEXT:   ^bb1:  // pred: ^bb0
# CHECK-NEXT:     %0 = "jlir.constant"() {value = #jlir.Int64} : () -> !jlir.DataType
# CHECK-NEXT:     %1 = "jlir.isa"(%arg1, %0) : (!jlir<"Union{Float64, Int64}">, !jlir.DataType) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%1)[^bb3, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb2:  // pred: ^bb1
# CHECK-NEXT:     %2 = "jlir.pi"(%arg1) : (!jlir<"Union{Float64, Int64}">) -> !jlir.Int64
# CHECK-NEXT:     %3 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %4 = "jlir.add_int"(%2, %3) : (!jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %5 = "jlir.pi"(%4) : (!jlir.Int64) -> !jlir<"Union{Nothing, Int64}">
# CHECK-NEXT:     "jlir.return"(%5) : (!jlir<"Union{Nothing, Int64}">) -> ()
# CHECK-NEXT:   ^bb3:  // pred: ^bb1
# CHECK-NEXT:     %6 = "jlir.constant"() {value = #jlir.nothing} : () -> !jlir.Nothing
# CHECK-NEXT:     %7 = "jlir.pi"(%6) : (!jlir.Nothing) -> !jlir<"Union{Nothing, Int64}">
# CHECK-NEXT:     "jlir.return"(%7) : (!jlir<"Union{Nothing, Int64}">) -> ()
# CHECK-NEXT:   }
# CHECK-NEXT: }
