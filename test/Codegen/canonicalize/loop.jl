# RUN: julia -e "import Brutus; Brutus.lit(:emit_optimized)" --startup-file=no %s 2>&1 | FileCheck %s

function loop(N)
    acc = 0
    for i in 1:N
        acc += i
    end
    return acc
end
emit(loop, Int64)



# CHECK: module  {
# CHECK-NEXT:   func nested @"Tuple{typeof(Main.loop), Int64}"(%arg0: !jlir<"typeof(Main.loop)">, %arg1: !jlir.Int64, %arg2: !jlir<"Union{Nothing, Tuple{Int64, Int64}}">, %arg3: !jlir.Int64, %arg4: !jlir.Int64) -> !jlir.Int64 attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     "jlir.goto"()[^bb1] : () -> ()
# CHECK-NEXT:   ^bb1:  // pred: ^bb0
# CHECK-NEXT:     %0 = "jlir.constant"() {value = #jlir<"#<intrinsic #29 sle_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %1 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %2 = "jlir.call"(%0, %1, %arg1) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%2)[^bb3, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb2:  // pred: ^bb1
# CHECK-NEXT:     "jlir.goto"(%arg1)[^bb4] : (!jlir.Int64) -> ()
# CHECK-NEXT:   ^bb3:  // pred: ^bb1
# CHECK-NEXT:     %3 = "jlir.constant"() {value = #jlir<"0">} : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%3)[^bb4] : (!jlir.Int64) -> ()
# CHECK-NEXT:   ^bb4(%4: !jlir.Int64):  // 2 preds: ^bb2, ^bb3
# CHECK-NEXT:     "jlir.goto"()[^bb5] : () -> ()
# CHECK-NEXT:   ^bb5:  // pred: ^bb4
# CHECK-NEXT:     "jlir.goto"()[^bb6] : () -> ()
# CHECK-NEXT:   ^bb6:  // pred: ^bb5
# CHECK-NEXT:     %5 = "jlir.constant"() {value = #jlir<"#<intrinsic #27 slt_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %6 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %7 = "jlir.call"(%5, %4, %6) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%7)[^bb8, ^bb7] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb7:  // pred: ^bb6
# CHECK-NEXT:     %8 = "jlir.constant"() {value = #jlir.nothing} : () -> !jlir.Nothing
# CHECK-NEXT:     %9 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     %10 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %11 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%9, %10, %11)[^bb9] : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb8:  // pred: ^bb6
# CHECK-NEXT:     %12 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK-NEXT:     %13 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %14 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%12, %13, %14)[^bb9] : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb9(%15: !jlir.Bool, %16: !jlir.Int64, %17: !jlir.Int64):  // 2 preds: ^bb7, ^bb8
# CHECK-NEXT:     %18 = "jlir.constant"() {value = #jlir<"#<intrinsic #43 not_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %19 = "jlir.call"(%18, %15) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     %20 = "jlir.constant"() {value = #jlir<"0">} : () -> !jlir.Int64
# CHECK-NEXT:     %21 = "jlir.constant"() {value = #jlir<"0">} : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.gotoifnot"(%19, %21, %16, %17, %20)[^bb15, ^bb10] {operand_segment_sizes = dense<[1, 1, 3]> : vector<3xi32>} : (!jlir.Bool, !jlir.Int64, !jlir.Int64, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb10(%22: !jlir.Int64, %23: !jlir.Int64, %24: !jlir.Int64):  // 2 preds: ^bb9, ^bb14
# CHECK-NEXT:     %25 = "jlir.constant"() {value = #jlir<"#<intrinsic #2 add_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %26 = "jlir.call"(%25, %24, %22) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %27 = "jlir.constant"() {value = #jlir<"===">} : () -> !jlir<"typeof(===)">
# CHECK-NEXT:     %28 = "jlir.call"(%27, %23, %4) : (!jlir<"typeof(===)">, !jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%28)[^bb12, ^bb11] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb11:  // pred: ^bb10
# CHECK-NEXT:     %29 = "jlir.constant"() {value = #jlir.nothing} : () -> !jlir.Nothing
# CHECK-NEXT:     %30 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %31 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %32 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%30, %31, %32)[^bb13] : (!jlir.Int64, !jlir.Int64, !jlir.Bool) -> ()
# CHECK-NEXT:   ^bb12:  // pred: ^bb10
# CHECK-NEXT:     %33 = "jlir.constant"() {value = #jlir<"#<intrinsic #2 add_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %34 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %35 = "jlir.call"(%33, %23, %34) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %36 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%35, %35, %36)[^bb13] : (!jlir.Int64, !jlir.Int64, !jlir.Bool) -> ()
# CHECK-NEXT:   ^bb13(%37: !jlir.Int64, %38: !jlir.Int64, %39: !jlir.Bool):  // 2 preds: ^bb11, ^bb12
# CHECK-NEXT:     %40 = "jlir.constant"() {value = #jlir<"#<intrinsic #43 not_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %41 = "jlir.call"(%40, %39) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%41, %26)[^bb15, ^bb14] {operand_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>} : (!jlir.Bool, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb14:  // pred: ^bb13
# CHECK-NEXT:     "jlir.goto"(%37, %38, %26)[^bb10] : (!jlir.Int64, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb15(%42: !jlir.Int64):  // 2 preds: ^bb9, ^bb13
# CHECK-NEXT:     "jlir.return"(%42) : (!jlir.Int64) -> ()
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK: module  {
# CHECK-NEXT:   func nested @"Tuple{typeof(Main.loop), Int64}"(%arg0: !jlir<"typeof(Main.loop)">, %arg1: !jlir.Int64, %arg2: !jlir<"Union{Nothing, Tuple{Int64, Int64}}">, %arg3: !jlir.Int64, %arg4: !jlir.Int64) -> !jlir.Int64 attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     "jlir.goto"()[^bb1] : () -> ()
# CHECK-NEXT:   ^bb1:  // pred: ^bb0
# CHECK-NEXT:     %0 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %1 = "jlir.sle_int"(%0, %arg1) : (!jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%1)[^bb3, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb2:  // pred: ^bb1
# CHECK-NEXT:     "jlir.goto"(%arg1)[^bb4] : (!jlir.Int64) -> ()
# CHECK-NEXT:   ^bb3:  // pred: ^bb1
# CHECK-NEXT:     %2 = "jlir.constant"() {value = #jlir<"0">} : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%2)[^bb4] : (!jlir.Int64) -> ()
# CHECK-NEXT:   ^bb4(%3: !jlir.Int64):  // 2 preds: ^bb2, ^bb3
# CHECK-NEXT:     "jlir.goto"()[^bb5] : () -> ()
# CHECK-NEXT:   ^bb5:  // pred: ^bb4
# CHECK-NEXT:     "jlir.goto"()[^bb6] : () -> ()
# CHECK-NEXT:   ^bb6:  // pred: ^bb5
# CHECK-NEXT:     %4 = "jlir.slt_int"(%3, %0) : (!jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%4)[^bb8, ^bb7] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb7:  // pred: ^bb6
# CHECK-NEXT:     %5 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     %6 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %7 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%5, %6, %7)[^bb9] : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb8:  // pred: ^bb6
# CHECK-NEXT:     %8 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%8, %0, %0)[^bb9] : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb9(%9: !jlir.Bool, %10: !jlir.Int64, %11: !jlir.Int64):  // 2 preds: ^bb7, ^bb8
# CHECK-NEXT:     %12 = "jlir.not_int"(%9) : (!jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     %13 = "jlir.constant"() {value = #jlir<"0">} : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.gotoifnot"(%12, %13, %10, %11, %13)[^bb15, ^bb10] {operand_segment_sizes = dense<[1, 1, 3]> : vector<3xi32>} : (!jlir.Bool, !jlir.Int64, !jlir.Int64, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb10(%14: !jlir.Int64, %15: !jlir.Int64, %16: !jlir.Int64):  // 2 preds: ^bb9, ^bb14
# CHECK-NEXT:     %17 = "jlir.add_int"(%16, %14) : (!jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %18 = "jlir.==="(%15, %3) : (!jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%18)[^bb12, ^bb11] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb11:  // pred: ^bb10
# CHECK-NEXT:     %19 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %20 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %21 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%19, %20, %21)[^bb13] : (!jlir.Int64, !jlir.Int64, !jlir.Bool) -> ()
# CHECK-NEXT:   ^bb12:  // pred: ^bb10
# CHECK-NEXT:     %22 = "jlir.add_int"(%15, %0) : (!jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %23 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%22, %22, %23)[^bb13] : (!jlir.Int64, !jlir.Int64, !jlir.Bool) -> ()
# CHECK-NEXT:   ^bb13(%24: !jlir.Int64, %25: !jlir.Int64, %26: !jlir.Bool):  // 2 preds: ^bb11, ^bb12
# CHECK-NEXT:     %27 = "jlir.not_int"(%26) : (!jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%27, %17)[^bb15, ^bb14] {operand_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>} : (!jlir.Bool, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb14:  // pred: ^bb13
# CHECK-NEXT:     "jlir.goto"(%24, %25, %17)[^bb10] : (!jlir.Int64, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb15(%28: !jlir.Int64):  // 2 preds: ^bb9, ^bb13
# CHECK-NEXT:     "jlir.return"(%28) : (!jlir.Int64) -> ()
# CHECK-NEXT:   }
# CHECK-NEXT: }
