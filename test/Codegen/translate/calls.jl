# RUN: julia -e "import Brutus; Brutus.lit(:emit_translated)" --startup-file=no %s 2>&1 | FileCheck %s

function calls()
    f = rand(Bool) ? (+) : (-)
    return f(1, 1)
end
emit(calls)



# CHECK: module {
# CHECK-NEXT:   func @"Tuple{typeof(Main.calls)}"(%arg0: !jlir<"typeof(Main.calls)">) -> !jlir.Int64 {
# CHECK-NEXT:     "jlir.goto"()[^bb1] : () -> ()
# CHECK-NEXT:   ^bb1:  // pred: ^bb0
# CHECK-NEXT:     %0 = "jlir.constant"() {value = #jlir<"typeof(Base.rand)()">} : () -> !jlir<"typeof(Base.rand)">
# CHECK-NEXT:     %1 = "jlir.constant"() {value = #jlir.Bool} : () -> !jlir.DataType
# CHECK-NEXT:     %2 = "jlir.invoke"(%0, %1) {methodInstance = #jlir<"rand(Type{Bool}) from rand(Type{X}) where {X}">} : (!jlir<"typeof(Base.rand)">, !jlir.DataType) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%2)[^bb3, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb2:  // pred: ^bb1
# CHECK-NEXT:     %3 = "jlir.constant"() {value = #jlir<"typeof(Base.:(+))()">} : () -> !jlir<"typeof(Base.:(+))">
# CHECK-NEXT:     %4 = "jlir.pi"(%3) : (!jlir<"typeof(Base.:(+))">) -> !jlir<"Union{typeof(Base.:(+)), typeof(Base.:(-))}">
# CHECK-NEXT:     "jlir.goto"(%4)[^bb4] : (!jlir<"Union{typeof(Base.:(+)), typeof(Base.:(-))}">) -> ()
# CHECK-NEXT:   ^bb3:  // pred: ^bb1
# CHECK-NEXT:     %5 = "jlir.constant"() {value = #jlir<"typeof(Base.:(-))()">} : () -> !jlir<"typeof(Base.:(-))">
# CHECK-NEXT:     %6 = "jlir.pi"(%5) : (!jlir<"typeof(Base.:(-))">) -> !jlir<"Union{typeof(Base.:(+)), typeof(Base.:(-))}">
# CHECK-NEXT:     "jlir.goto"(%6)[^bb4] : (!jlir<"Union{typeof(Base.:(+)), typeof(Base.:(-))}">) -> ()
# CHECK-NEXT:   ^bb4(%7: !jlir<"Union{typeof(Base.:(+)), typeof(Base.:(-))}">):  // 2 preds: ^bb2, ^bb3
# CHECK-NEXT:     %8 = "jlir.constant"() {value = #jlir<"typeof(isa)()">} : () -> !jlir<"typeof(isa)">
# CHECK-NEXT:     %9 = "jlir.constant"() {value = #jlir<"typeof(Base.:(+))">} : () -> !jlir.DataType
# CHECK-NEXT:     %10 = "jlir.call"(%8, %7, %9) : (!jlir<"typeof(isa)">, !jlir<"Union{typeof(Base.:(+)), typeof(Base.:(-))}">, !jlir.DataType) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%10)[^bb6, ^bb5] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb5:  // pred: ^bb4
# CHECK-NEXT:     %11 = "jlir.constant"() {value = #jlir<"#<intrinsic #2 add_int>">} : () -> !jlir.Core.IntrinsicFunction
# CHECK-NEXT:     %12 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %13 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %14 = "jlir.call"(%11, %12, %13) : (!jlir.Core.IntrinsicFunction, !jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%14)[^bb9] : (!jlir.Int64) -> ()
# CHECK-NEXT:   ^bb6:  // pred: ^bb4
# CHECK-NEXT:     %15 = "jlir.constant"() {value = #jlir<"typeof(isa)()">} : () -> !jlir<"typeof(isa)">
# CHECK-NEXT:     %16 = "jlir.constant"() {value = #jlir<"typeof(Base.:(-))">} : () -> !jlir.DataType
# CHECK-NEXT:     %17 = "jlir.call"(%15, %7, %16) : (!jlir<"typeof(isa)">, !jlir<"Union{typeof(Base.:(+)), typeof(Base.:(-))}">, !jlir.DataType) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%17)[^bb8, ^bb7] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb7:  // pred: ^bb6
# CHECK-NEXT:     %18 = "jlir.constant"() {value = #jlir<"#<intrinsic #3 sub_int>">} : () -> !jlir.Core.IntrinsicFunction
# CHECK-NEXT:     %19 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %20 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %21 = "jlir.call"(%18, %19, %20) : (!jlir.Core.IntrinsicFunction, !jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%21)[^bb9] : (!jlir.Int64) -> ()
# CHECK-NEXT:   ^bb8:  // pred: ^bb6
# CHECK-NEXT:     %22 = "jlir.constant"() {value = #jlir<"typeof(throw)()">} : () -> !jlir<"typeof(throw)">
# CHECK-NEXT:     %23 = "jlir.constant"() {value = #jlir<"ErrorException("fatal error in type inference (type bound)")">} : () -> !jlir.ErrorException
# CHECK-NEXT:     %24 = "jlir.call"(%22, %23) : (!jlir<"typeof(throw)">, !jlir.ErrorException) -> !jlir<"Union{}">
# CHECK-NEXT:     %25 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.return"(%25) : (!jlir.Int64) -> ()
# CHECK-NEXT:   ^bb9(%26: !jlir.Int64):  // 2 preds: ^bb5, ^bb7
# CHECK-NEXT:     "jlir.return"(%26) : (!jlir.Int64) -> ()
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK: error: lowering to LLVM dialect failed
