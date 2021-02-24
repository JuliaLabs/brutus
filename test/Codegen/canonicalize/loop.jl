# RUN: julia -e "import Brutus; Brutus.lit(:emit_optimized)" --startup-file=no %s 2>&1 | FileCheck %s

function loop(N)
    acc = 0
    for i in 1:N
        acc += i
    end
    return acc
end
emit(loop, Int64)



# CHECK:   func nested @"Tuple{typeof(Main.loop), Int64}"(%arg0: !jlir<"typeof(Main.loop)">, %arg1: !jlir.Int64) -> !jlir.Int64 attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     "jlir.goto"()[^bb1] : () -> ()
# CHECK-NEXT:   ^bb1:  // pred: ^bb0
# CHECK-NEXT:     %0 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %1 = "jlir.sle_int"(%0, %arg1) : (!jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     %2 = "jlir.constant"() {value = #jlir<"0">} : () -> !jlir.Int64
# CHECK-NEXT:     %3 = "jlir.ifelse"(%1, %arg1, %2) : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %4 = "jlir.slt_int"(%3, %0) : (!jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%4)[^bb3, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb2:  // pred: ^bb1
# CHECK-NEXT:     %5 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     %6 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %7 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%5, %6, %7)[^bb4] : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb3:  // pred: ^bb1
# CHECK-NEXT:     %8 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%8, %0, %0)[^bb4] : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb4(%9: !jlir.Bool, %10: !jlir.Int64, %11: !jlir.Int64):  // 2 preds: ^bb2, ^bb3
# CHECK-NEXT:     %12 = "jlir.not_int"(%9) : (!jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%12, %2, %10, %11, %2)[^bb10, ^bb5] {operand_segment_sizes = dense<[1, 1, 3]> : vector<3xi32>} : (!jlir.Bool, !jlir.Int64, !jlir.Int64, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb5(%13: !jlir.Int64, %14: !jlir.Int64, %15: !jlir.Int64):  // 2 preds: ^bb4, ^bb9
# CHECK-NEXT:     %16 = "jlir.add_int"(%15, %13) : (!jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %17 = "jlir.==="(%14, %3) : (!jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%17)[^bb7, ^bb6] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb6:  // pred: ^bb5
# CHECK-NEXT:     %18 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %19 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %20 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%18, %19, %20)[^bb8] : (!jlir.Int64, !jlir.Int64, !jlir.Bool) -> ()
# CHECK-NEXT:   ^bb7:  // pred: ^bb5
# CHECK-NEXT:     %21 = "jlir.add_int"(%14, %0) : (!jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %22 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%21, %21, %22)[^bb8] : (!jlir.Int64, !jlir.Int64, !jlir.Bool) -> ()
# CHECK-NEXT:   ^bb8(%23: !jlir.Int64, %24: !jlir.Int64, %25: !jlir.Bool):  // 2 preds: ^bb6, ^bb7
# CHECK-NEXT:     %26 = "jlir.not_int"(%25) : (!jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%26, %16)[^bb10, ^bb9] {operand_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>} : (!jlir.Bool, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb9:  // pred: ^bb8
# CHECK-NEXT:     "jlir.goto"(%23, %24, %16)[^bb5] : (!jlir.Int64, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb10(%27: !jlir.Int64):  // 2 preds: ^bb4, ^bb8
# CHECK-NEXT:     "jlir.return"(%27) : (!jlir.Int64) -> ()
# CHECK-NEXT:   }
# CHECK-NEXT: }
