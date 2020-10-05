# RUN: julia -e "import Brutus; Brutus.lit(:emit_translated)" --startup-file=no %s 2>&1 | FileCheck %s



function calls()
    f = rand(Bool) ? (+) : (-)
    return f(1, 1)
end
emit(calls)

# CHECK: func @"Tuple{typeof(Main.calls)}"(%arg0: !jlir<"typeof(Main.calls)">) -> !jlir.Any
# CHECK:   "jlir.goto"()[^bb1] : () -> ()
# CHECK: ^bb1:  // pred: ^bb0
# CHECK:   %0 = "jlir.constant"() {value = #jlir<"typeof(Base.rand)()">} : () -> !jlir<"typeof(Base.rand)">
# CHECK:   %1 = "jlir.constant"() {value = #jlir.Bool} : () -> !jlir.DataType
# CHECK:   %2 = "jlir.invoke"(%0, %1) {methodInstance = #jlir<"rand(Type{Bool})">} : (!jlir<"typeof(Base.rand)">, !jlir.DataType) -> !jlir.Bool
# CHECK:   "jlir.gotoifnot"(%2)[^bb3, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK: ^bb2:  // pred: ^bb1
# CHECK:   %3 = "jlir.constant"() {value = #jlir<"typeof(Base.:(+))()">} : () -> !jlir<"typeof(Base.:(+))">
# CHECK:   %4 = "jlir.pi"(%3) : (!jlir<"typeof(Base.:(+))">) -> !jlir<"Union{typeof(Base.:(+)), typeof(Base.:(-))}">
# CHECK:   "jlir.goto"(%4)[^bb4] : (!jlir<"Union{typeof(Base.:(+)), typeof(Base.:(-))}">) -> ()
# CHECK: ^bb3:  // pred: ^bb1
# CHECK:   %5 = "jlir.constant"() {value = #jlir<"typeof(Base.:(-))()">} : () -> !jlir<"typeof(Base.:(-))">
# CHECK:   %6 = "jlir.pi"(%5) : (!jlir<"typeof(Base.:(-))">) -> !jlir<"Union{typeof(Base.:(+)), typeof(Base.:(-))}">
# CHECK:   "jlir.goto"(%6)[^bb4] : (!jlir<"Union{typeof(Base.:(+)), typeof(Base.:(-))}">) -> ()
# CHECK: ^bb4(%7: !jlir<"Union{typeof(Base.:(+)), typeof(Base.:(-))}">):
# CHECK:   %8 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK:   %9 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK:   %10 = "jlir.call"(%7, %8, %9) : (!jlir<"Union{typeof(Base.:(+)), typeof(Base.:(-))}">, !jlir.Int64, !jlir.Int64) -> !jlir.Any
# CHECK:   "jlir.return"(%10) : (!jlir.Any) -> ()
