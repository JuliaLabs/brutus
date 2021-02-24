# RUN: julia -e "import Brutus; Brutus.lit(:emit_optimized)" --startup-file=no %s 2>&1 | FileCheck %s

function sum(A)
    acc = zero(eltype(A))
    for i in 1:prod(size(A))
        acc += A[i]
    end
    return acc
end

emit(sum, Matrix{Float64})



# CHECK:   func @"Tuple{typeof(Main.sum), Array{Float64, 2}}"(%arg0: !jlir<"typeof(Main.sum)">, %arg1: !jlir<"Array{Float64, 2}">) -> !jlir.Float64 attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     "jlir.goto"()[^bb1] : () -> ()
# CHECK-NEXT:   ^bb1:  // pred: ^bb0
# CHECK-NEXT:     %0 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %1 = "jlir.arraysize"(%arg1, %0) : (!jlir<"Array{Float64, 2}">, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %2 = "jlir.constant"() {value = #jlir<"2">} : () -> !jlir.Int64
# CHECK-NEXT:     %3 = "jlir.arraysize"(%arg1, %2) : (!jlir<"Array{Float64, 2}">, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %4 = "jlir.mul_int"(%1, %3) : (!jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %5 = "jlir.sle_int"(%0, %4) : (!jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     %6 = "jlir.constant"() {value = #jlir<"0">} : () -> !jlir.Int64
# CHECK-NEXT:     %7 = "jlir.ifelse"(%5, %4, %6) : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %8 = "jlir.slt_int"(%7, %0) : (!jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%8)[^bb3, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb2:  // pred: ^bb1
# CHECK-NEXT:     %9 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     %10 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %11 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%9, %10, %11)[^bb4] : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb3:  // pred: ^bb1
# CHECK-NEXT:     %12 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%12, %0, %0)[^bb4] : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb4(%13: !jlir.Bool, %14: !jlir.Int64, %15: !jlir.Int64):  // 2 preds: ^bb2, ^bb3
# CHECK-NEXT:     %16 = "jlir.not_int"(%13) : (!jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     %17 = "jlir.constant"() {value = #jlir<"0">} : () -> !jlir.Float64
# CHECK-NEXT:     "jlir.gotoifnot"(%16, %17, %14, %15, %17)[^bb10, ^bb5] {operand_segment_sizes = dense<[1, 1, 3]> : vector<3xi32>} : (!jlir.Bool, !jlir.Float64, !jlir.Int64, !jlir.Int64, !jlir.Float64) -> ()
# CHECK-NEXT:   ^bb5(%18: !jlir.Int64, %19: !jlir.Int64, %20: !jlir.Float64):  // 2 preds: ^bb4, ^bb9
# CHECK-NEXT:     %21 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     %22 = "jlir.arrayref"(%21, %arg1, %18) : (!jlir.Bool, !jlir<"Array{Float64, 2}">, !jlir.Int64) -> !jlir.Float64
# CHECK-NEXT:     %23 = "jlir.add_float"(%20, %22) : (!jlir.Float64, !jlir.Float64) -> !jlir.Float64
# CHECK-NEXT:     %24 = "jlir.==="(%19, %7) : (!jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%24)[^bb7, ^bb6] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb6:  // pred: ^bb5
# CHECK-NEXT:     %25 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %26 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%25, %26, %21)[^bb8] : (!jlir.Int64, !jlir.Int64, !jlir.Bool) -> ()
# CHECK-NEXT:   ^bb7:  // pred: ^bb5
# CHECK-NEXT:     %27 = "jlir.add_int"(%19, %0) : (!jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %28 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%27, %27, %28)[^bb8] : (!jlir.Int64, !jlir.Int64, !jlir.Bool) -> ()
# CHECK-NEXT:   ^bb8(%29: !jlir.Int64, %30: !jlir.Int64, %31: !jlir.Bool):  // 2 preds: ^bb6, ^bb7
# CHECK-NEXT:     %32 = "jlir.not_int"(%31) : (!jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%32, %23)[^bb10, ^bb9] {operand_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>} : (!jlir.Bool, !jlir.Float64) -> ()
# CHECK-NEXT:   ^bb9:  // pred: ^bb8
# CHECK-NEXT:     "jlir.goto"(%29, %30, %23)[^bb5] : (!jlir.Int64, !jlir.Int64, !jlir.Float64) -> ()
# CHECK-NEXT:   ^bb10(%33: !jlir.Float64):  // 2 preds: ^bb4, ^bb8
# CHECK-NEXT:     "jlir.return"(%33) : (!jlir.Float64) -> ()
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK: error: lowering to LLVM dialect failed
