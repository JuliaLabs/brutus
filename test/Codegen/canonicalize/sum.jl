# RUN: julia -e "import Brutus; Brutus.lit(:emit_optimized)" --startup-file=no %s 2>&1 | FileCheck %s

function sum(A)
    acc = zero(eltype(A))
    for i in 1:prod(size(A))
        acc += A[i]
    end
    return acc
end

emit(sum, Matrix{Float64})



# CHECK: module  {
# CHECK-NEXT:   func nested @"Tuple{typeof(Main.sum), Array{Float64, 2}}"(%arg0: !jlir<"typeof(Main.sum)">, %arg1: !jlir<"Array{Float64, 2}">, %arg2: !jlir<"Union{Nothing, Tuple{Int64, Int64}}">, %arg3: !jlir.Float64, %arg4: !jlir.Int64) -> !jlir.Float64 attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     "jlir.goto"()[^bb1] : () -> ()
# CHECK-NEXT:   ^bb1:  // pred: ^bb0
# CHECK-NEXT:     %0 = "jlir.constant"() {value = #jlir.Core.arraysize} : () -> !jlir<"typeof(Core.arraysize)">
# CHECK-NEXT:     %1 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %2 = "jlir.call"(%0, %arg1, %1) : (!jlir<"typeof(Core.arraysize)">, !jlir<"Array{Float64, 2}">, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %3 = "jlir.constant"() {value = #jlir.Core.arraysize} : () -> !jlir<"typeof(Core.arraysize)">
# CHECK-NEXT:     %4 = "jlir.constant"() {value = #jlir<"2">} : () -> !jlir.Int64
# CHECK-NEXT:     %5 = "jlir.call"(%3, %arg1, %4) : (!jlir<"typeof(Core.arraysize)">, !jlir<"Array{Float64, 2}">, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %6 = "jlir.constant"() {value = #jlir<"#<intrinsic #4 mul_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %7 = "jlir.call"(%6, %2, %5) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %8 = "jlir.constant"() {value = #jlir<"#<intrinsic #29 sle_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %9 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %10 = "jlir.call"(%8, %9, %7) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%10)[^bb3, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb2:  // pred: ^bb1
# CHECK-NEXT:     "jlir.goto"(%7)[^bb4] : (!jlir.Int64) -> ()
# CHECK-NEXT:   ^bb3:  // pred: ^bb1
# CHECK-NEXT:     %11 = "jlir.constant"() {value = #jlir<"0">} : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%11)[^bb4] : (!jlir.Int64) -> ()
# CHECK-NEXT:   ^bb4(%12: !jlir.Int64):  // 2 preds: ^bb2, ^bb3
# CHECK-NEXT:     "jlir.goto"()[^bb5] : () -> ()
# CHECK-NEXT:   ^bb5:  // pred: ^bb4
# CHECK-NEXT:     "jlir.goto"()[^bb6] : () -> ()
# CHECK-NEXT:   ^bb6:  // pred: ^bb5
# CHECK-NEXT:     %13 = "jlir.constant"() {value = #jlir<"#<intrinsic #27 slt_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %14 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %15 = "jlir.call"(%13, %12, %14) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%15)[^bb8, ^bb7] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb7:  // pred: ^bb6
# CHECK-NEXT:     %16 = "jlir.constant"() {value = #jlir.nothing} : () -> !jlir.Nothing
# CHECK-NEXT:     %17 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     %18 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %19 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%17, %18, %19)[^bb9] : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb8:  // pred: ^bb6
# CHECK-NEXT:     %20 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK-NEXT:     %21 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %22 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%20, %21, %22)[^bb9] : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb9(%23: !jlir.Bool, %24: !jlir.Int64, %25: !jlir.Int64):  // 2 preds: ^bb7, ^bb8
# CHECK-NEXT:     %26 = "jlir.constant"() {value = #jlir<"#<intrinsic #43 not_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %27 = "jlir.call"(%26, %23) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     %28 = "jlir.constant"() {value = #jlir<"0">} : () -> !jlir.Float64
# CHECK-NEXT:     %29 = "jlir.constant"() {value = #jlir<"0">} : () -> !jlir.Float64
# CHECK-NEXT:     "jlir.gotoifnot"(%27, %29, %24, %25, %28)[^bb15, ^bb10] {operand_segment_sizes = dense<[1, 1, 3]> : vector<3xi32>} : (!jlir.Bool, !jlir.Float64, !jlir.Int64, !jlir.Int64, !jlir.Float64) -> ()
# CHECK-NEXT:   ^bb10(%30: !jlir.Int64, %31: !jlir.Int64, %32: !jlir.Float64):  // 2 preds: ^bb9, ^bb14
# CHECK-NEXT:     %33 = "jlir.constant"() {value = #jlir.Core.arrayref} : () -> !jlir<"typeof(Core.arrayref)">
# CHECK-NEXT:     %34 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     %35 = "jlir.call"(%33, %34, %arg1, %30) : (!jlir<"typeof(Core.arrayref)">, !jlir.Bool, !jlir<"Array{Float64, 2}">, !jlir.Int64) -> !jlir.Float64
# CHECK-NEXT:     %36 = "jlir.constant"() {value = #jlir<"#<intrinsic #12 add_float>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %37 = "jlir.call"(%36, %32, %35) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Float64, !jlir.Float64) -> !jlir.Float64
# CHECK-NEXT:     %38 = "jlir.constant"() {value = #jlir<"===">} : () -> !jlir<"typeof(===)">
# CHECK-NEXT:     %39 = "jlir.call"(%38, %31, %12) : (!jlir<"typeof(===)">, !jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%39)[^bb12, ^bb11] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb11:  // pred: ^bb10
# CHECK-NEXT:     %40 = "jlir.constant"() {value = #jlir.nothing} : () -> !jlir.Nothing
# CHECK-NEXT:     %41 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %42 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %43 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%41, %42, %43)[^bb13] : (!jlir.Int64, !jlir.Int64, !jlir.Bool) -> ()
# CHECK-NEXT:   ^bb12:  // pred: ^bb10
# CHECK-NEXT:     %44 = "jlir.constant"() {value = #jlir<"#<intrinsic #2 add_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %45 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %46 = "jlir.call"(%44, %31, %45) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %47 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%46, %46, %47)[^bb13] : (!jlir.Int64, !jlir.Int64, !jlir.Bool) -> ()
# CHECK-NEXT:   ^bb13(%48: !jlir.Int64, %49: !jlir.Int64, %50: !jlir.Bool):  // 2 preds: ^bb11, ^bb12
# CHECK-NEXT:     %51 = "jlir.constant"() {value = #jlir<"#<intrinsic #43 not_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %52 = "jlir.call"(%51, %50) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%52, %37)[^bb15, ^bb14] {operand_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>} : (!jlir.Bool, !jlir.Float64) -> ()
# CHECK-NEXT:   ^bb14:  // pred: ^bb13
# CHECK-NEXT:     "jlir.goto"(%48, %49, %37)[^bb10] : (!jlir.Int64, !jlir.Int64, !jlir.Float64) -> ()
# CHECK-NEXT:   ^bb15(%53: !jlir.Float64):  // 2 preds: ^bb9, ^bb13
# CHECK-NEXT:     "jlir.return"(%53) : (!jlir.Float64) -> ()
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK: module  {
# CHECK-NEXT:   func nested @"Tuple{typeof(Main.sum), Array{Float64, 2}}"(%arg0: !jlir<"typeof(Main.sum)">, %arg1: !jlir<"Array{Float64, 2}">, %arg2: !jlir<"Union{Nothing, Tuple{Int64, Int64}}">, %arg3: !jlir.Float64, %arg4: !jlir.Int64) -> !jlir.Float64 attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     "jlir.goto"()[^bb1] : () -> ()
# CHECK-NEXT:   ^bb1:  // pred: ^bb0
# CHECK-NEXT:     %0 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %1 = "jlir.arraysize"(%arg1, %0) : (!jlir<"Array{Float64, 2}">, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %2 = "jlir.constant"() {value = #jlir<"2">} : () -> !jlir.Int64
# CHECK-NEXT:     %3 = "jlir.arraysize"(%arg1, %2) : (!jlir<"Array{Float64, 2}">, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %4 = "jlir.mul_int"(%1, %3) : (!jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %5 = "jlir.sle_int"(%0, %4) : (!jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%5)[^bb3, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb2:  // pred: ^bb1
# CHECK-NEXT:     "jlir.goto"(%4)[^bb4] : (!jlir.Int64) -> ()
# CHECK-NEXT:   ^bb3:  // pred: ^bb1
# CHECK-NEXT:     %6 = "jlir.constant"() {value = #jlir<"0">} : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%6)[^bb4] : (!jlir.Int64) -> ()
# CHECK-NEXT:   ^bb4(%7: !jlir.Int64):  // 2 preds: ^bb2, ^bb3
# CHECK-NEXT:     "jlir.goto"()[^bb5] : () -> ()
# CHECK-NEXT:   ^bb5:  // pred: ^bb4
# CHECK-NEXT:     "jlir.goto"()[^bb6] : () -> ()
# CHECK-NEXT:   ^bb6:  // pred: ^bb5
# CHECK-NEXT:     %8 = "jlir.slt_int"(%7, %0) : (!jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%8)[^bb8, ^bb7] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb7:  // pred: ^bb6
# CHECK-NEXT:     %9 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     %10 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %11 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%9, %10, %11)[^bb9] : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb8:  // pred: ^bb6
# CHECK-NEXT:     %12 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%12, %0, %0)[^bb9] : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb9(%13: !jlir.Bool, %14: !jlir.Int64, %15: !jlir.Int64):  // 2 preds: ^bb7, ^bb8
# CHECK-NEXT:     %16 = "jlir.not_int"(%13) : (!jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     %17 = "jlir.constant"() {value = #jlir<"0">} : () -> !jlir.Float64
# CHECK-NEXT:     "jlir.gotoifnot"(%16, %17, %14, %15, %17)[^bb15, ^bb10] {operand_segment_sizes = dense<[1, 1, 3]> : vector<3xi32>} : (!jlir.Bool, !jlir.Float64, !jlir.Int64, !jlir.Int64, !jlir.Float64) -> ()
# CHECK-NEXT:   ^bb10(%18: !jlir.Int64, %19: !jlir.Int64, %20: !jlir.Float64):  // 2 preds: ^bb9, ^bb14
# CHECK-NEXT:     %21 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     %22 = "jlir.arrayref"(%21, %arg1, %18) : (!jlir.Bool, !jlir<"Array{Float64, 2}">, !jlir.Int64) -> !jlir.Float64
# CHECK-NEXT:     %23 = "jlir.add_float"(%20, %22) : (!jlir.Float64, !jlir.Float64) -> !jlir.Float64
# CHECK-NEXT:     %24 = "jlir.==="(%19, %7) : (!jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%24)[^bb12, ^bb11] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb11:  // pred: ^bb10
# CHECK-NEXT:     %25 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %26 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%25, %26, %21)[^bb13] : (!jlir.Int64, !jlir.Int64, !jlir.Bool) -> ()
# CHECK-NEXT:   ^bb12:  // pred: ^bb10
# CHECK-NEXT:     %27 = "jlir.add_int"(%19, %0) : (!jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %28 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%27, %27, %28)[^bb13] : (!jlir.Int64, !jlir.Int64, !jlir.Bool) -> ()
# CHECK-NEXT:   ^bb13(%29: !jlir.Int64, %30: !jlir.Int64, %31: !jlir.Bool):  // 2 preds: ^bb11, ^bb12
# CHECK-NEXT:     %32 = "jlir.not_int"(%31) : (!jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%32, %23)[^bb15, ^bb14] {operand_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>} : (!jlir.Bool, !jlir.Float64) -> ()
# CHECK-NEXT:   ^bb14:  // pred: ^bb13
# CHECK-NEXT:     "jlir.goto"(%29, %30, %23)[^bb10] : (!jlir.Int64, !jlir.Int64, !jlir.Float64) -> ()
# CHECK-NEXT:   ^bb15(%33: !jlir.Float64):  // 2 preds: ^bb9, ^bb13
# CHECK-NEXT:     "jlir.return"(%33) : (!jlir.Float64) -> ()
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK: error: lowering to LLVM dialect failed
# CHECK-NEXT: error: module verification failed
