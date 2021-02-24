# RUN: julia -e "import Brutus; Brutus.lit(:emit_optimized)" --startup-file=no %s 2>&1 | FileCheck %s

# Intentionally bad code
function matmul!(C, A, B)
    for i in 1:size(A, 1)
        for k in 1:size(A, 2)
            for j in 1:size(B, 2)
                C[i, j] += A[i, k] * B[k, j]
            end
        end
    end
   return nothing
end

emit(matmul!, Matrix{Float64}, Matrix{Float64}, Matrix{Float64})



# CHECK:   func @"Tuple{typeof(Main.matmul!), Array{Float64, 2}, Array{Float64, 2}, Array{Float64, 2}}"(%arg0: !jlir<"typeof(Main.matmul!)">, %arg1: !jlir<"Array{Float64, 2}">, %arg2: !jlir<"Array{Float64, 2}">, %arg3: !jlir<"Array{Float64, 2}">) -> !jlir.Nothing attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     "jlir.goto"()[^bb1] : () -> ()
# CHECK-NEXT:   ^bb1:  // pred: ^bb0
# CHECK-NEXT:     %0 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %1 = "jlir.arraysize"(%arg2, %0) : (!jlir<"Array{Float64, 2}">, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %2 = "jlir.sle_int"(%0, %1) : (!jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     %3 = "jlir.constant"() {value = #jlir<"0">} : () -> !jlir.Int64
# CHECK-NEXT:     %4 = "jlir.ifelse"(%2, %1, %3) : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %5 = "jlir.slt_int"(%4, %0) : (!jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%5)[^bb3, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb2:  // pred: ^bb1
# CHECK-NEXT:     %6 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     %7 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %8 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%6, %7, %8)[^bb4] : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb3:  // pred: ^bb1
# CHECK-NEXT:     %9 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%9, %0, %0)[^bb4] : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb4(%10: !jlir.Bool, %11: !jlir.Int64, %12: !jlir.Int64):  // 2 preds: ^bb2, ^bb3
# CHECK-NEXT:     %13 = "jlir.not_int"(%10) : (!jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%13, %11, %12)[^bb28, ^bb5] {operand_segment_sizes = dense<[1, 0, 2]> : vector<3xi32>} : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb5(%14: !jlir.Int64, %15: !jlir.Int64):  // 2 preds: ^bb4, ^bb27
# CHECK-NEXT:     %16 = "jlir.constant"() {value = #jlir<"2">} : () -> !jlir.Int64
# CHECK-NEXT:     %17 = "jlir.arraysize"(%arg2, %16) : (!jlir<"Array{Float64, 2}">, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %18 = "jlir.sle_int"(%0, %17) : (!jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     %19 = "jlir.ifelse"(%18, %17, %3) : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %20 = "jlir.slt_int"(%19, %0) : (!jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%20)[^bb7, ^bb6] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb6:  // pred: ^bb5
# CHECK-NEXT:     %21 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     %22 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %23 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%21, %22, %23)[^bb8] : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb7:  // pred: ^bb5
# CHECK-NEXT:     %24 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%24, %0, %0)[^bb8] : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb8(%25: !jlir.Bool, %26: !jlir.Int64, %27: !jlir.Int64):  // 2 preds: ^bb6, ^bb7
# CHECK-NEXT:     %28 = "jlir.not_int"(%25) : (!jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%28, %26, %27)[^bb23, ^bb9] {operand_segment_sizes = dense<[1, 0, 2]> : vector<3xi32>} : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb9(%29: !jlir.Int64, %30: !jlir.Int64):  // 2 preds: ^bb8, ^bb22
# CHECK-NEXT:     %31 = "jlir.arraysize"(%arg3, %16) : (!jlir<"Array{Float64, 2}">, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %32 = "jlir.sle_int"(%0, %31) : (!jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     %33 = "jlir.ifelse"(%32, %31, %3) : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %34 = "jlir.slt_int"(%33, %0) : (!jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%34)[^bb11, ^bb10] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb10:  // pred: ^bb9
# CHECK-NEXT:     %35 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     %36 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %37 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%35, %36, %37)[^bb12] : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb11:  // pred: ^bb9
# CHECK-NEXT:     %38 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%38, %0, %0)[^bb12] : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb12(%39: !jlir.Bool, %40: !jlir.Int64, %41: !jlir.Int64):  // 2 preds: ^bb10, ^bb11
# CHECK-NEXT:     %42 = "jlir.not_int"(%39) : (!jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%42, %40, %41)[^bb18, ^bb13] {operand_segment_sizes = dense<[1, 0, 2]> : vector<3xi32>} : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb13(%43: !jlir.Int64, %44: !jlir.Int64):  // 2 preds: ^bb12, ^bb17
# CHECK-NEXT:     %45 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     %46 = "jlir.arrayref"(%45, %arg1, %14, %43) : (!jlir.Bool, !jlir<"Array{Float64, 2}">, !jlir.Int64, !jlir.Int64) -> !jlir.Float64
# CHECK-NEXT:     %47 = "jlir.arrayref"(%45, %arg2, %14, %29) : (!jlir.Bool, !jlir<"Array{Float64, 2}">, !jlir.Int64, !jlir.Int64) -> !jlir.Float64
# CHECK-NEXT:     %48 = "jlir.arrayref"(%45, %arg3, %29, %43) : (!jlir.Bool, !jlir<"Array{Float64, 2}">, !jlir.Int64, !jlir.Int64) -> !jlir.Float64
# CHECK-NEXT:     %49 = "jlir.mul_float"(%47, %48) : (!jlir.Float64, !jlir.Float64) -> !jlir.Float64
# CHECK-NEXT:     %50 = "jlir.add_float"(%46, %49) : (!jlir.Float64, !jlir.Float64) -> !jlir.Float64
# CHECK-NEXT:     %51 = "jlir.arrayset"(%45, %arg1, %50, %14, %43) : (!jlir.Bool, !jlir<"Array{Float64, 2}">, !jlir.Float64, !jlir.Int64, !jlir.Int64) -> !jlir<"Array{Float64, 2}">
# CHECK-NEXT:     %52 = "jlir.==="(%44, %33) : (!jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%52)[^bb15, ^bb14] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb14:  // pred: ^bb13
# CHECK-NEXT:     %53 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %54 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%53, %54, %45)[^bb16] : (!jlir.Int64, !jlir.Int64, !jlir.Bool) -> ()
# CHECK-NEXT:   ^bb15:  // pred: ^bb13
# CHECK-NEXT:     %55 = "jlir.add_int"(%44, %0) : (!jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %56 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%55, %55, %56)[^bb16] : (!jlir.Int64, !jlir.Int64, !jlir.Bool) -> ()
# CHECK-NEXT:   ^bb16(%57: !jlir.Int64, %58: !jlir.Int64, %59: !jlir.Bool):  // 2 preds: ^bb14, ^bb15
# CHECK-NEXT:     %60 = "jlir.not_int"(%59) : (!jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%60)[^bb18, ^bb17] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb17:  // pred: ^bb16
# CHECK-NEXT:     "jlir.goto"(%57, %58)[^bb13] : (!jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb18:  // 2 preds: ^bb12, ^bb16
# CHECK-NEXT:     %61 = "jlir.==="(%30, %19) : (!jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%61)[^bb20, ^bb19] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb19:  // pred: ^bb18
# CHECK-NEXT:     %62 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %63 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %64 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%62, %63, %64)[^bb21] : (!jlir.Int64, !jlir.Int64, !jlir.Bool) -> ()
# CHECK-NEXT:   ^bb20:  // pred: ^bb18
# CHECK-NEXT:     %65 = "jlir.add_int"(%30, %0) : (!jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %66 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%65, %65, %66)[^bb21] : (!jlir.Int64, !jlir.Int64, !jlir.Bool) -> ()
# CHECK-NEXT:   ^bb21(%67: !jlir.Int64, %68: !jlir.Int64, %69: !jlir.Bool):  // 2 preds: ^bb19, ^bb20
# CHECK-NEXT:     %70 = "jlir.not_int"(%69) : (!jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%70)[^bb23, ^bb22] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb22:  // pred: ^bb21
# CHECK-NEXT:     "jlir.goto"(%67, %68)[^bb9] : (!jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb23:  // 2 preds: ^bb8, ^bb21
# CHECK-NEXT:     %71 = "jlir.==="(%15, %4) : (!jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%71)[^bb25, ^bb24] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb24:  // pred: ^bb23
# CHECK-NEXT:     %72 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %73 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %74 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%72, %73, %74)[^bb26] : (!jlir.Int64, !jlir.Int64, !jlir.Bool) -> ()
# CHECK-NEXT:   ^bb25:  // pred: ^bb23
# CHECK-NEXT:     %75 = "jlir.add_int"(%15, %0) : (!jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %76 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%75, %75, %76)[^bb26] : (!jlir.Int64, !jlir.Int64, !jlir.Bool) -> ()
# CHECK-NEXT:   ^bb26(%77: !jlir.Int64, %78: !jlir.Int64, %79: !jlir.Bool):  // 2 preds: ^bb24, ^bb25
# CHECK-NEXT:     %80 = "jlir.not_int"(%79) : (!jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%80)[^bb28, ^bb27] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb27:  // pred: ^bb26
# CHECK-NEXT:     "jlir.goto"(%77, %78)[^bb5] : (!jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb28:  // 2 preds: ^bb4, ^bb26
# CHECK-NEXT:     %81 = "jlir.constant"() {value = #jlir.nothing} : () -> !jlir.Nothing
# CHECK-NEXT:     "jlir.return"(%81) : (!jlir.Nothing) -> ()
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK: error: lowering to LLVM dialect failed
