# RUN: julia -e "import Brutus; Brutus.lit(:emit_optimized)" --startup-file=no %s 2>&1 | FileCheck %s

function loop(N)
    acc = 0
    for i in 1:N
        acc += i
    end
    return acc
end
emit(loop, Int64)



# CHECK: Core.MethodMatch(Tuple{typeof(Main.Main.loop), Int64}, svec(), loop(N) in Main.Main at /home/mccoy/Dev/brutus/test/Codegen/canonicalize/loop.jl:3, true)after translating to MLIR in JLIR dialect:module  {
# CHECK-NEXT:   func nested @"Tuple{typeof(Main.loop), Int64}"(%arg0: !jlir<"typeof(Main.loop)">, %arg1: !jlir.Int64) -> !jlir.Int64 attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     "jlir.goto"()[^bb1] : () -> ()
# CHECK-NEXT:   ^bb1:  // pred: ^bb0
# CHECK-NEXT:     %0 = "jlir.constant"() {value = #jlir<"#<intrinsic #29 sle_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %1 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %2 = "jlir.call"(%0, %1, %arg1) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     %3 = "jlir.constant"() {value = #jlir.ifelse} : () -> !jlir<"typeof(ifelse)">
# CHECK-NEXT:     %4 = "jlir.constant"() {value = #jlir<"0">} : () -> !jlir.Int64
# CHECK-NEXT:     %5 = "jlir.call"(%3, %2, %arg1, %4) : (!jlir<"typeof(ifelse)">, !jlir.Bool, !jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %6 = "jlir.constant"() {value = #jlir<"#<intrinsic #27 slt_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %7 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %8 = "jlir.call"(%6, %5, %7) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%8)[^bb3, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb2:  // pred: ^bb1
# CHECK-NEXT:     %9 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     %10 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %11 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%9, %10, %11)[^bb4] : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb3:  // pred: ^bb1
# CHECK-NEXT:     %12 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK-NEXT:     %13 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %14 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%12, %13, %14)[^bb4] : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb4(%15: !jlir.Bool, %16: !jlir.Int64, %17: !jlir.Int64):  // 2 preds: ^bb2, ^bb3
# CHECK-NEXT:     %18 = "jlir.constant"() {value = #jlir<"#<intrinsic #44 not_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %19 = "jlir.call"(%18, %15) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     %20 = "jlir.constant"() {value = #jlir<"0">} : () -> !jlir.Int64
# CHECK-NEXT:     %21 = "jlir.constant"() {value = #jlir<"0">} : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.gotoifnot"(%19, %21, %16, %17, %20)[^bb10, ^bb5] {operand_segment_sizes = dense<[1, 1, 3]> : vector<3xi32>} : (!jlir.Bool, !jlir.Int64, !jlir.Int64, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb5(%22: !jlir.Int64, %23: !jlir.Int64, %24: !jlir.Int64):  // 2 preds: ^bb4, ^bb9
# CHECK-NEXT:     %25 = "jlir.constant"() {value = #jlir<"#<intrinsic #2 add_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %26 = "jlir.call"(%25, %24, %22) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %27 = "jlir.constant"() {value = #jlir<"===">} : () -> !jlir<"typeof(===)">
# CHECK-NEXT:     %28 = "jlir.call"(%27, %23, %5) : (!jlir<"typeof(===)">, !jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%28)[^bb7, ^bb6] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb6:  // pred: ^bb5
# CHECK-NEXT:     %29 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %30 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %31 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%29, %30, %31)[^bb8] : (!jlir.Int64, !jlir.Int64, !jlir.Bool) -> ()
# CHECK-NEXT:   ^bb7:  // pred: ^bb5
# CHECK-NEXT:     %32 = "jlir.constant"() {value = #jlir<"#<intrinsic #2 add_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %33 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %34 = "jlir.call"(%32, %23, %33) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %35 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%34, %34, %35)[^bb8] : (!jlir.Int64, !jlir.Int64, !jlir.Bool) -> ()
# CHECK-NEXT:   ^bb8(%36: !jlir.Int64, %37: !jlir.Int64, %38: !jlir.Bool):  // 2 preds: ^bb6, ^bb7
# CHECK-NEXT:     %39 = "jlir.constant"() {value = #jlir<"#<intrinsic #44 not_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %40 = "jlir.call"(%39, %38) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%40, %26)[^bb10, ^bb9] {operand_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>} : (!jlir.Bool, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb9:  // pred: ^bb8
# CHECK-NEXT:     "jlir.goto"(%36, %37, %26)[^bb5] : (!jlir.Int64, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb10(%41: !jlir.Int64):  // 2 preds: ^bb4, ^bb8
# CHECK-NEXT:     "jlir.return"(%41) : (!jlir.Int64) -> ()
# CHECK-NEXT:   }
# CHECK-NEXT: }

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
