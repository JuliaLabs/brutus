# RUN: julia -e "import Brutus; Brutus.lit(:emit_translated)" --startup-file=no %s 2>&1 | FileCheck %s




function loop(N)
    acc = 1
    for i in 1:N
        acc += i
    end
    return acc
end
emit(loop, Int64)
# CHECK: func @"Tuple{typeof(Main.loop), Int64}"(%arg0: !jlir<"typeof(Main.loop)">, %arg1: !jlir.Int64) -> !jlir.Int64
# CHECK:   "jlir.goto"()[^bb1] : () -> ()
# CHECK: ^bb1:
# CHECK:   %0 = "jlir.constant"() {value = #jlir<"#<intrinsic #29 sle_int>">} : () -> !jlir.Core.IntrinsicFunction
# CHECK:   %1 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK:   %2 = "jlir.call"(%0, %1, %arg1) : (!jlir.Core.IntrinsicFunction, !jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK:   %3 = "jlir.constant"() {value = #jlir<"typeof(ifelse)()">} : () -> !jlir<"typeof(ifelse)">
# CHECK:   %4 = "jlir.constant"() {value = #jlir<"0">} : () -> !jlir.Int64
# CHECK:   %5 = "jlir.call"(%3, %2, %arg1, %4) : (!jlir<"typeof(ifelse)">, !jlir.Bool, !jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK:   %6 = "jlir.constant"() {value = #jlir<"#<intrinsic #27 slt_int>">} : () -> !jlir.Core.IntrinsicFunction
# CHECK:   %7 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK:   %8 = "jlir.call"(%6, %5, %7) : (!jlir.Core.IntrinsicFunction, !jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK:   "jlir.gotoifnot"(%8)[^bb3, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK: ^bb2:
# CHECK:   %9 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK:   %10 = "jlir.undef"() : () -> !jlir.Int64
# CHECK:   %11 = "jlir.undef"() : () -> !jlir.Int64
# CHECK:   "jlir.goto"(%9, %10, %11)[^bb4] : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK: ^bb3:
# CHECK:   %12 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK:   %13 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK:   %14 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK:   "jlir.goto"(%12, %13, %14)[^bb4] : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK: ^bb4(%15: !jlir.Bool, %16: !jlir.Int64, %17: !jlir.Int64):
# CHECK:   %18 = "jlir.constant"() {value = #jlir<"#<intrinsic #44 not_int>">} : () -> !jlir.Core.IntrinsicFunction
# CHECK:   %19 = "jlir.call"(%18, %15) : (!jlir.Core.IntrinsicFunction, !jlir.Bool) -> !jlir.Bool
# CHECK:   %20 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK:   %21 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK:   "jlir.gotoifnot"(%19, %21, %20, %16, %17)[^bb10, ^bb5] {operand_segment_sizes = dense<[1, 1, 3]> : vector<3xi32>} : (!jlir.Bool, !jlir.Int64, !jlir.Int64, !jlir.Int64, !jlir.Int64) -> ()
# CHECK: ^bb5(%22: !jlir.Int64, %23: !jlir.Int64, %24: !jlir.Int64):
# CHECK:   %25 = "jlir.constant"() {value = #jlir<"#<intrinsic #2 add_int>">} : () -> !jlir.Core.IntrinsicFunction
# CHECK:   %26 = "jlir.call"(%25, %22, %23) : (!jlir.Core.IntrinsicFunction, !jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK:   %27 = "jlir.constant"() {value = #jlir<"typeof(===)()">} : () -> !jlir<"typeof(===)">
# CHECK:   %28 = "jlir.call"(%27, %24, %5) : (!jlir<"typeof(===)">, !jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK:   "jlir.gotoifnot"(%28)[^bb7, ^bb6] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK: ^bb6:
# CHECK:   %29 = "jlir.undef"() : () -> !jlir.Int64
# CHECK:   %30 = "jlir.undef"() : () -> !jlir.Int64
# CHECK:   %31 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK:   "jlir.goto"(%29, %30, %31)[^bb8] : (!jlir.Int64, !jlir.Int64, !jlir.Bool) -> ()
# CHECK: ^bb7:
# CHECK:   %32 = "jlir.constant"() {value = #jlir<"#<intrinsic #2 add_int>">} : () -> !jlir.Core.IntrinsicFunction
# CHECK:   %33 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK:   %34 = "jlir.call"(%32, %24, %33) : (!jlir.Core.IntrinsicFunction, !jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK:   %35 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK:   "jlir.goto"(%34, %34, %35)[^bb8] : (!jlir.Int64, !jlir.Int64, !jlir.Bool) -> ()
# CHECK: ^bb8(%36: !jlir.Int64, %37: !jlir.Int64, %38: !jlir.Bool):
# CHECK:   %39 = "jlir.constant"() {value = #jlir<"#<intrinsic #44 not_int>">} : () -> !jlir.Core.IntrinsicFunction
# CHECK:   %40 = "jlir.call"(%39, %38) : (!jlir.Core.IntrinsicFunction, !jlir.Bool) -> !jlir.Bool
# CHECK:   "jlir.gotoifnot"(%40, %26)[^bb10, ^bb9] {operand_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>} : (!jlir.Bool, !jlir.Int64) -> ()
# CHECK: ^bb9:
# CHECK:   "jlir.goto"(%26, %36, %37)[^bb5] : (!jlir.Int64, !jlir.Int64, !jlir.Int64) -> ()
# CHECK: ^bb10(%41: !jlir.Int64):
# CHECK:   "jlir.return"(%41) : (!jlir.Int64) -> ()
