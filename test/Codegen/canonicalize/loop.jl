# RUN: julia -e "import Brutus; Brutus.lit(:emit_optimized)" --startup-file=no %s 2>&1 | FileCheck %s

function loop(N)
    acc = 0
    for i in 1:N
        acc += i
    end
    return acc
end
emit(loop, Int64)
# CHECK: func @"Tuple{typeof(Main.loop), Int64}"(%arg0: !jlir<"typeof(Main.loop)">, %arg1: !jlir.Int64) -> !jlir.Int64
# CHECK:   "jlir.goto"()[^bb1] : () -> ()
# CHECK: ^bb1:
# CHECK:   %0 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK:   %1 = "jlir.sle_int"(%0, %arg1) : (!jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK:   %2 = "jlir.constant"() {value = #jlir<"0">} : () -> !jlir.Int64
# CHECK:   %3 = "jlir.ifelse"(%1, %arg1, %2) : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK:   %4 = "jlir.slt_int"(%3, %0) : (!jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK:   "jlir.gotoifnot"(%4)[^bb3, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK: ^bb2:
# CHECK:   %5 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK:   %6 = "jlir.undef"() : () -> !jlir.Int64
# CHECK:   %7 = "jlir.undef"() : () -> !jlir.Int64
# CHECK:   "jlir.goto"(%5, %6, %7)[^bb4] : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK: ^bb3:
# CHECK:   %8 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK:   "jlir.goto"(%8, %0, %0)[^bb4] : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK: ^bb4(%9: !jlir.Bool, %10: !jlir.Int64, %11: !jlir.Int64):
# CHECK:   %12 = "jlir.not_int"(%9) : (!jlir.Bool) -> !jlir.Bool
# CHECK:   "jlir.gotoifnot"(%12, %2, %2, %10, %11)[^bb10, ^bb5] {operand_segment_sizes = dense<[1, 1, 3]> : vector<3xi32>} : (!jlir.Bool, !jlir.Int64, !jlir.Int64, !jlir.Int64, !jlir.Int64) -> ()
# CHECK: ^bb5(%13: !jlir.Int64, %14: !jlir.Int64, %15: !jlir.Int64):
# CHECK:   %16 = "jlir.add_int"(%13, %14) : (!jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK:   %17 = "jlir.==="(%15, %3) : (!jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK:   "jlir.gotoifnot"(%17)[^bb7, ^bb6] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK: ^bb6:
# CHECK:   %18 = "jlir.undef"() : () -> !jlir.Int64
# CHECK:   %19 = "jlir.undef"() : () -> !jlir.Int64
# CHECK:   %20 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK:   "jlir.goto"(%18, %19, %20)[^bb8] : (!jlir.Int64, !jlir.Int64, !jlir.Bool) -> ()
# CHECK: ^bb7:
# CHECK:   %21 = "jlir.add_int"(%15, %0) : (!jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK:   %22 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK:   "jlir.goto"(%21, %21, %22)[^bb8] : (!jlir.Int64, !jlir.Int64, !jlir.Bool) -> ()
# CHECK: ^bb8(%23: !jlir.Int64, %24: !jlir.Int64, %25: !jlir.Bool):
# CHECK:   %26 = "jlir.not_int"(%25) : (!jlir.Bool) -> !jlir.Bool
# CHECK:   "jlir.gotoifnot"(%26, %16)[^bb10, ^bb9] {operand_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>} : (!jlir.Bool, !jlir.Int64) -> ()
# CHECK: ^bb9:
# CHECK:   "jlir.goto"(%16, %23, %24)[^bb5] : (!jlir.Int64, !jlir.Int64, !jlir.Int64) -> ()
# CHECK: ^bb10(%27: !jlir.Int64):
# CHECK:   "jlir.return"(%27) : (!jlir.Int64) -> ()
