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



# CHECK: module  {
# CHECK-NEXT:   func nested @"Tuple{typeof(Main.matmul!), Array{Float64, 2}, Array{Float64, 2}, Array{Float64, 2}}"(%arg0: !jlir<"typeof(Main.matmul!)">, %arg1: !jlir<"Array{Float64, 2}">, %arg2: !jlir<"Array{Float64, 2}">, %arg3: !jlir<"Array{Float64, 2}">, %arg4: !jlir<"Union{Nothing, Tuple{Int64, Int64}}">, %arg5: !jlir<"Union{Nothing, Tuple{Int64, Int64}}">, %arg6: !jlir.Int64, %arg7: !jlir<"Union{Nothing, Tuple{Int64, Int64}}">, %arg8: !jlir.Int64, %arg9: !jlir.Int64) -> !jlir.Nothing attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     "jlir.goto"()[^bb1] : () -> ()
# CHECK-NEXT:   ^bb1:  // pred: ^bb0
# CHECK-NEXT:     %0 = "jlir.constant"() {value = #jlir.Core.arraysize} : () -> !jlir<"typeof(Core.arraysize)">
# CHECK-NEXT:     %1 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %2 = "jlir.call"(%0, %arg2, %1) : (!jlir<"typeof(Core.arraysize)">, !jlir<"Array{Float64, 2}">, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %3 = "jlir.constant"() {value = #jlir<"#<intrinsic #29 sle_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %4 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %5 = "jlir.call"(%3, %4, %2) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%5)[^bb3, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb2:  // pred: ^bb1
# CHECK-NEXT:     "jlir.goto"(%2)[^bb4] : (!jlir.Int64) -> ()
# CHECK-NEXT:   ^bb3:  // pred: ^bb1
# CHECK-NEXT:     %6 = "jlir.constant"() {value = #jlir<"0">} : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%6)[^bb4] : (!jlir.Int64) -> ()
# CHECK-NEXT:   ^bb4(%7: !jlir.Int64):  // 2 preds: ^bb2, ^bb3
# CHECK-NEXT:     "jlir.goto"()[^bb5] : () -> ()
# CHECK-NEXT:   ^bb5:  // pred: ^bb4
# CHECK-NEXT:     "jlir.goto"()[^bb6] : () -> ()
# CHECK-NEXT:   ^bb6:  // pred: ^bb5
# CHECK-NEXT:     %8 = "jlir.constant"() {value = #jlir<"#<intrinsic #27 slt_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %9 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %10 = "jlir.call"(%8, %7, %9) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%10)[^bb8, ^bb7] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb7:  // pred: ^bb6
# CHECK-NEXT:     %11 = "jlir.constant"() {value = #jlir.nothing} : () -> !jlir.Nothing
# CHECK-NEXT:     %12 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     %13 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %14 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%12, %13, %14)[^bb9] : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb8:  // pred: ^bb6
# CHECK-NEXT:     %15 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK-NEXT:     %16 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %17 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%15, %16, %17)[^bb9] : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb9(%18: !jlir.Bool, %19: !jlir.Int64, %20: !jlir.Int64):  // 2 preds: ^bb7, ^bb8
# CHECK-NEXT:     %21 = "jlir.constant"() {value = #jlir<"#<intrinsic #43 not_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %22 = "jlir.call"(%21, %18) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%22, %19, %20)[^bb43, ^bb10] {operand_segment_sizes = dense<[1, 0, 2]> : vector<3xi32>} : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb10(%23: !jlir.Int64, %24: !jlir.Int64):  // 2 preds: ^bb9, ^bb42
# CHECK-NEXT:     %25 = "jlir.constant"() {value = #jlir.Core.arraysize} : () -> !jlir<"typeof(Core.arraysize)">
# CHECK-NEXT:     %26 = "jlir.constant"() {value = #jlir<"2">} : () -> !jlir.Int64
# CHECK-NEXT:     %27 = "jlir.call"(%25, %arg2, %26) : (!jlir<"typeof(Core.arraysize)">, !jlir<"Array{Float64, 2}">, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %28 = "jlir.constant"() {value = #jlir<"#<intrinsic #29 sle_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %29 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %30 = "jlir.call"(%28, %29, %27) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%30)[^bb12, ^bb11] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb11:  // pred: ^bb10
# CHECK-NEXT:     "jlir.goto"(%27)[^bb13] : (!jlir.Int64) -> ()
# CHECK-NEXT:   ^bb12:  // pred: ^bb10
# CHECK-NEXT:     %31 = "jlir.constant"() {value = #jlir<"0">} : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%31)[^bb13] : (!jlir.Int64) -> ()
# CHECK-NEXT:   ^bb13(%32: !jlir.Int64):  // 2 preds: ^bb11, ^bb12
# CHECK-NEXT:     "jlir.goto"()[^bb14] : () -> ()
# CHECK-NEXT:   ^bb14:  // pred: ^bb13
# CHECK-NEXT:     "jlir.goto"()[^bb15] : () -> ()
# CHECK-NEXT:   ^bb15:  // pred: ^bb14
# CHECK-NEXT:     %33 = "jlir.constant"() {value = #jlir<"#<intrinsic #27 slt_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %34 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %35 = "jlir.call"(%33, %32, %34) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%35)[^bb17, ^bb16] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb16:  // pred: ^bb15
# CHECK-NEXT:     %36 = "jlir.constant"() {value = #jlir.nothing} : () -> !jlir.Nothing
# CHECK-NEXT:     %37 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     %38 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %39 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%37, %38, %39)[^bb18] : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb17:  // pred: ^bb15
# CHECK-NEXT:     %40 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK-NEXT:     %41 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %42 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%40, %41, %42)[^bb18] : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb18(%43: !jlir.Bool, %44: !jlir.Int64, %45: !jlir.Int64):  // 2 preds: ^bb16, ^bb17
# CHECK-NEXT:     %46 = "jlir.constant"() {value = #jlir<"#<intrinsic #43 not_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %47 = "jlir.call"(%46, %43) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%47, %44, %45)[^bb38, ^bb19] {operand_segment_sizes = dense<[1, 0, 2]> : vector<3xi32>} : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb19(%48: !jlir.Int64, %49: !jlir.Int64):  // 2 preds: ^bb18, ^bb37
# CHECK-NEXT:     %50 = "jlir.constant"() {value = #jlir.Core.arraysize} : () -> !jlir<"typeof(Core.arraysize)">
# CHECK-NEXT:     %51 = "jlir.constant"() {value = #jlir<"2">} : () -> !jlir.Int64
# CHECK-NEXT:     %52 = "jlir.call"(%50, %arg3, %51) : (!jlir<"typeof(Core.arraysize)">, !jlir<"Array{Float64, 2}">, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %53 = "jlir.constant"() {value = #jlir<"#<intrinsic #29 sle_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %54 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %55 = "jlir.call"(%53, %54, %52) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%55)[^bb21, ^bb20] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb20:  // pred: ^bb19
# CHECK-NEXT:     "jlir.goto"(%52)[^bb22] : (!jlir.Int64) -> ()
# CHECK-NEXT:   ^bb21:  // pred: ^bb19
# CHECK-NEXT:     %56 = "jlir.constant"() {value = #jlir<"0">} : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%56)[^bb22] : (!jlir.Int64) -> ()
# CHECK-NEXT:   ^bb22(%57: !jlir.Int64):  // 2 preds: ^bb20, ^bb21
# CHECK-NEXT:     "jlir.goto"()[^bb23] : () -> ()
# CHECK-NEXT:   ^bb23:  // pred: ^bb22
# CHECK-NEXT:     "jlir.goto"()[^bb24] : () -> ()
# CHECK-NEXT:   ^bb24:  // pred: ^bb23
# CHECK-NEXT:     %58 = "jlir.constant"() {value = #jlir<"#<intrinsic #27 slt_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %59 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %60 = "jlir.call"(%58, %57, %59) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%60)[^bb26, ^bb25] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb25:  // pred: ^bb24
# CHECK-NEXT:     %61 = "jlir.constant"() {value = #jlir.nothing} : () -> !jlir.Nothing
# CHECK-NEXT:     %62 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     %63 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %64 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%62, %63, %64)[^bb27] : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb26:  // pred: ^bb24
# CHECK-NEXT:     %65 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK-NEXT:     %66 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %67 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%65, %66, %67)[^bb27] : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb27(%68: !jlir.Bool, %69: !jlir.Int64, %70: !jlir.Int64):  // 2 preds: ^bb25, ^bb26
# CHECK-NEXT:     %71 = "jlir.constant"() {value = #jlir<"#<intrinsic #43 not_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %72 = "jlir.call"(%71, %68) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%72, %69, %70)[^bb33, ^bb28] {operand_segment_sizes = dense<[1, 0, 2]> : vector<3xi32>} : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb28(%73: !jlir.Int64, %74: !jlir.Int64):  // 2 preds: ^bb27, ^bb32
# CHECK-NEXT:     %75 = "jlir.constant"() {value = #jlir.Core.arrayref} : () -> !jlir<"typeof(Core.arrayref)">
# CHECK-NEXT:     %76 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     %77 = "jlir.call"(%75, %76, %arg1, %23, %73) : (!jlir<"typeof(Core.arrayref)">, !jlir.Bool, !jlir<"Array{Float64, 2}">, !jlir.Int64, !jlir.Int64) -> !jlir.Float64
# CHECK-NEXT:     %78 = "jlir.constant"() {value = #jlir.Core.arrayref} : () -> !jlir<"typeof(Core.arrayref)">
# CHECK-NEXT:     %79 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     %80 = "jlir.call"(%78, %79, %arg2, %23, %48) : (!jlir<"typeof(Core.arrayref)">, !jlir.Bool, !jlir<"Array{Float64, 2}">, !jlir.Int64, !jlir.Int64) -> !jlir.Float64
# CHECK-NEXT:     %81 = "jlir.constant"() {value = #jlir.Core.arrayref} : () -> !jlir<"typeof(Core.arrayref)">
# CHECK-NEXT:     %82 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     %83 = "jlir.call"(%81, %82, %arg3, %48, %73) : (!jlir<"typeof(Core.arrayref)">, !jlir.Bool, !jlir<"Array{Float64, 2}">, !jlir.Int64, !jlir.Int64) -> !jlir.Float64
# CHECK-NEXT:     %84 = "jlir.constant"() {value = #jlir<"#<intrinsic #14 mul_float>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %85 = "jlir.call"(%84, %80, %83) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Float64, !jlir.Float64) -> !jlir.Float64
# CHECK-NEXT:     %86 = "jlir.constant"() {value = #jlir<"#<intrinsic #12 add_float>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %87 = "jlir.call"(%86, %77, %85) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Float64, !jlir.Float64) -> !jlir.Float64
# CHECK-NEXT:     %88 = "jlir.constant"() {value = #jlir.Core.arrayset} : () -> !jlir<"typeof(Core.arrayset)">
# CHECK-NEXT:     %89 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     %90 = "jlir.call"(%88, %89, %arg1, %87, %23, %73) : (!jlir<"typeof(Core.arrayset)">, !jlir.Bool, !jlir<"Array{Float64, 2}">, !jlir.Float64, !jlir.Int64, !jlir.Int64) -> !jlir<"Array{Float64, 2}">
# CHECK-NEXT:     %91 = "jlir.constant"() {value = #jlir<"===">} : () -> !jlir<"typeof(===)">
# CHECK-NEXT:     %92 = "jlir.call"(%91, %74, %57) : (!jlir<"typeof(===)">, !jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%92)[^bb30, ^bb29] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb29:  // pred: ^bb28
# CHECK-NEXT:     %93 = "jlir.constant"() {value = #jlir.nothing} : () -> !jlir.Nothing
# CHECK-NEXT:     %94 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %95 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %96 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%94, %95, %96)[^bb31] : (!jlir.Int64, !jlir.Int64, !jlir.Bool) -> ()
# CHECK-NEXT:   ^bb30:  // pred: ^bb28
# CHECK-NEXT:     %97 = "jlir.constant"() {value = #jlir<"#<intrinsic #2 add_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %98 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %99 = "jlir.call"(%97, %74, %98) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %100 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%99, %99, %100)[^bb31] : (!jlir.Int64, !jlir.Int64, !jlir.Bool) -> ()
# CHECK-NEXT:   ^bb31(%101: !jlir.Int64, %102: !jlir.Int64, %103: !jlir.Bool):  // 2 preds: ^bb29, ^bb30
# CHECK-NEXT:     %104 = "jlir.constant"() {value = #jlir<"#<intrinsic #43 not_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %105 = "jlir.call"(%104, %103) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%105)[^bb33, ^bb32] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb32:  // pred: ^bb31
# CHECK-NEXT:     "jlir.goto"(%101, %102)[^bb28] : (!jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb33:  // 2 preds: ^bb27, ^bb31
# CHECK-NEXT:     %106 = "jlir.constant"() {value = #jlir<"===">} : () -> !jlir<"typeof(===)">
# CHECK-NEXT:     %107 = "jlir.call"(%106, %49, %32) : (!jlir<"typeof(===)">, !jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%107)[^bb35, ^bb34] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb34:  // pred: ^bb33
# CHECK-NEXT:     %108 = "jlir.constant"() {value = #jlir.nothing} : () -> !jlir.Nothing
# CHECK-NEXT:     %109 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %110 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %111 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%109, %110, %111)[^bb36] : (!jlir.Int64, !jlir.Int64, !jlir.Bool) -> ()
# CHECK-NEXT:   ^bb35:  // pred: ^bb33
# CHECK-NEXT:     %112 = "jlir.constant"() {value = #jlir<"#<intrinsic #2 add_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %113 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %114 = "jlir.call"(%112, %49, %113) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %115 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%114, %114, %115)[^bb36] : (!jlir.Int64, !jlir.Int64, !jlir.Bool) -> ()
# CHECK-NEXT:   ^bb36(%116: !jlir.Int64, %117: !jlir.Int64, %118: !jlir.Bool):  // 2 preds: ^bb34, ^bb35
# CHECK-NEXT:     %119 = "jlir.constant"() {value = #jlir<"#<intrinsic #43 not_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %120 = "jlir.call"(%119, %118) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%120)[^bb38, ^bb37] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb37:  // pred: ^bb36
# CHECK-NEXT:     "jlir.goto"(%116, %117)[^bb19] : (!jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb38:  // 2 preds: ^bb18, ^bb36
# CHECK-NEXT:     %121 = "jlir.constant"() {value = #jlir<"===">} : () -> !jlir<"typeof(===)">
# CHECK-NEXT:     %122 = "jlir.call"(%121, %24, %7) : (!jlir<"typeof(===)">, !jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%122)[^bb40, ^bb39] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb39:  // pred: ^bb38
# CHECK-NEXT:     %123 = "jlir.constant"() {value = #jlir.nothing} : () -> !jlir.Nothing
# CHECK-NEXT:     %124 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %125 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %126 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%124, %125, %126)[^bb41] : (!jlir.Int64, !jlir.Int64, !jlir.Bool) -> ()
# CHECK-NEXT:   ^bb40:  // pred: ^bb38
# CHECK-NEXT:     %127 = "jlir.constant"() {value = #jlir<"#<intrinsic #2 add_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %128 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %129 = "jlir.call"(%127, %24, %128) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %130 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%129, %129, %130)[^bb41] : (!jlir.Int64, !jlir.Int64, !jlir.Bool) -> ()
# CHECK-NEXT:   ^bb41(%131: !jlir.Int64, %132: !jlir.Int64, %133: !jlir.Bool):  // 2 preds: ^bb39, ^bb40
# CHECK-NEXT:     %134 = "jlir.constant"() {value = #jlir<"#<intrinsic #43 not_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %135 = "jlir.call"(%134, %133) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%135)[^bb43, ^bb42] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb42:  // pred: ^bb41
# CHECK-NEXT:     "jlir.goto"(%131, %132)[^bb10] : (!jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb43:  // 2 preds: ^bb9, ^bb41
# CHECK-NEXT:     %136 = "jlir.constant"() {value = #jlir.nothing} : () -> !jlir.Nothing
# CHECK-NEXT:     "jlir.return"(%136) : (!jlir.Nothing) -> ()
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK: module  {
# CHECK-NEXT:   func nested @"Tuple{typeof(Main.matmul!), Array{Float64, 2}, Array{Float64, 2}, Array{Float64, 2}}"(%arg0: !jlir<"typeof(Main.matmul!)">, %arg1: !jlir<"Array{Float64, 2}">, %arg2: !jlir<"Array{Float64, 2}">, %arg3: !jlir<"Array{Float64, 2}">, %arg4: !jlir<"Union{Nothing, Tuple{Int64, Int64}}">, %arg5: !jlir<"Union{Nothing, Tuple{Int64, Int64}}">, %arg6: !jlir.Int64, %arg7: !jlir<"Union{Nothing, Tuple{Int64, Int64}}">, %arg8: !jlir.Int64, %arg9: !jlir.Int64) -> !jlir.Nothing attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     "jlir.goto"()[^bb1] : () -> ()
# CHECK-NEXT:   ^bb1:  // pred: ^bb0
# CHECK-NEXT:     %0 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %1 = "jlir.arraysize"(%arg2, %0) : (!jlir<"Array{Float64, 2}">, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %2 = "jlir.sle_int"(%0, %1) : (!jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%2)[^bb3, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb2:  // pred: ^bb1
# CHECK-NEXT:     "jlir.goto"(%1)[^bb4] : (!jlir.Int64) -> ()
# CHECK-NEXT:   ^bb3:  // pred: ^bb1
# CHECK-NEXT:     %3 = "jlir.constant"() {value = #jlir<"0">} : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%3)[^bb4] : (!jlir.Int64) -> ()
# CHECK-NEXT:   ^bb4(%4: !jlir.Int64):  // 2 preds: ^bb2, ^bb3
# CHECK-NEXT:     "jlir.goto"()[^bb5] : () -> ()
# CHECK-NEXT:   ^bb5:  // pred: ^bb4
# CHECK-NEXT:     "jlir.goto"()[^bb6] : () -> ()
# CHECK-NEXT:   ^bb6:  // pred: ^bb5
# CHECK-NEXT:     %5 = "jlir.slt_int"(%4, %0) : (!jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%5)[^bb8, ^bb7] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb7:  // pred: ^bb6
# CHECK-NEXT:     %6 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     %7 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %8 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%6, %7, %8)[^bb9] : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb8:  // pred: ^bb6
# CHECK-NEXT:     %9 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%9, %0, %0)[^bb9] : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb9(%10: !jlir.Bool, %11: !jlir.Int64, %12: !jlir.Int64):  // 2 preds: ^bb7, ^bb8
# CHECK-NEXT:     %13 = "jlir.not_int"(%10) : (!jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%13, %11, %12)[^bb43, ^bb10] {operand_segment_sizes = dense<[1, 0, 2]> : vector<3xi32>} : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb10(%14: !jlir.Int64, %15: !jlir.Int64):  // 2 preds: ^bb9, ^bb42
# CHECK-NEXT:     %16 = "jlir.constant"() {value = #jlir<"2">} : () -> !jlir.Int64
# CHECK-NEXT:     %17 = "jlir.arraysize"(%arg2, %16) : (!jlir<"Array{Float64, 2}">, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %18 = "jlir.sle_int"(%0, %17) : (!jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%18)[^bb12, ^bb11] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb11:  // pred: ^bb10
# CHECK-NEXT:     "jlir.goto"(%17)[^bb13] : (!jlir.Int64) -> ()
# CHECK-NEXT:   ^bb12:  // pred: ^bb10
# CHECK-NEXT:     %19 = "jlir.constant"() {value = #jlir<"0">} : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%19)[^bb13] : (!jlir.Int64) -> ()
# CHECK-NEXT:   ^bb13(%20: !jlir.Int64):  // 2 preds: ^bb11, ^bb12
# CHECK-NEXT:     "jlir.goto"()[^bb14] : () -> ()
# CHECK-NEXT:   ^bb14:  // pred: ^bb13
# CHECK-NEXT:     "jlir.goto"()[^bb15] : () -> ()
# CHECK-NEXT:   ^bb15:  // pred: ^bb14
# CHECK-NEXT:     %21 = "jlir.slt_int"(%20, %0) : (!jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%21)[^bb17, ^bb16] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb16:  // pred: ^bb15
# CHECK-NEXT:     %22 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     %23 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %24 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%22, %23, %24)[^bb18] : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb17:  // pred: ^bb15
# CHECK-NEXT:     %25 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%25, %0, %0)[^bb18] : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb18(%26: !jlir.Bool, %27: !jlir.Int64, %28: !jlir.Int64):  // 2 preds: ^bb16, ^bb17
# CHECK-NEXT:     %29 = "jlir.not_int"(%26) : (!jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%29, %27, %28)[^bb38, ^bb19] {operand_segment_sizes = dense<[1, 0, 2]> : vector<3xi32>} : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb19(%30: !jlir.Int64, %31: !jlir.Int64):  // 2 preds: ^bb18, ^bb37
# CHECK-NEXT:     %32 = "jlir.arraysize"(%arg3, %16) : (!jlir<"Array{Float64, 2}">, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %33 = "jlir.sle_int"(%0, %32) : (!jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%33)[^bb21, ^bb20] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb20:  // pred: ^bb19
# CHECK-NEXT:     "jlir.goto"(%32)[^bb22] : (!jlir.Int64) -> ()
# CHECK-NEXT:   ^bb21:  // pred: ^bb19
# CHECK-NEXT:     %34 = "jlir.constant"() {value = #jlir<"0">} : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%34)[^bb22] : (!jlir.Int64) -> ()
# CHECK-NEXT:   ^bb22(%35: !jlir.Int64):  // 2 preds: ^bb20, ^bb21
# CHECK-NEXT:     "jlir.goto"()[^bb23] : () -> ()
# CHECK-NEXT:   ^bb23:  // pred: ^bb22
# CHECK-NEXT:     "jlir.goto"()[^bb24] : () -> ()
# CHECK-NEXT:   ^bb24:  // pred: ^bb23
# CHECK-NEXT:     %36 = "jlir.slt_int"(%35, %0) : (!jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%36)[^bb26, ^bb25] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb25:  // pred: ^bb24
# CHECK-NEXT:     %37 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     %38 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %39 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%37, %38, %39)[^bb27] : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb26:  // pred: ^bb24
# CHECK-NEXT:     %40 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%40, %0, %0)[^bb27] : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb27(%41: !jlir.Bool, %42: !jlir.Int64, %43: !jlir.Int64):  // 2 preds: ^bb25, ^bb26
# CHECK-NEXT:     %44 = "jlir.not_int"(%41) : (!jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%44, %42, %43)[^bb33, ^bb28] {operand_segment_sizes = dense<[1, 0, 2]> : vector<3xi32>} : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb28(%45: !jlir.Int64, %46: !jlir.Int64):  // 2 preds: ^bb27, ^bb32
# CHECK-NEXT:     %47 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     %48 = "jlir.arrayref"(%47, %arg1, %14, %45) : (!jlir.Bool, !jlir<"Array{Float64, 2}">, !jlir.Int64, !jlir.Int64) -> !jlir.Float64
# CHECK-NEXT:     %49 = "jlir.arrayref"(%47, %arg2, %14, %30) : (!jlir.Bool, !jlir<"Array{Float64, 2}">, !jlir.Int64, !jlir.Int64) -> !jlir.Float64
# CHECK-NEXT:     %50 = "jlir.arrayref"(%47, %arg3, %30, %45) : (!jlir.Bool, !jlir<"Array{Float64, 2}">, !jlir.Int64, !jlir.Int64) -> !jlir.Float64
# CHECK-NEXT:     %51 = "jlir.mul_float"(%49, %50) : (!jlir.Float64, !jlir.Float64) -> !jlir.Float64
# CHECK-NEXT:     %52 = "jlir.add_float"(%48, %51) : (!jlir.Float64, !jlir.Float64) -> !jlir.Float64
# CHECK-NEXT:     %53 = "jlir.arrayset"(%47, %arg1, %52, %14, %45) : (!jlir.Bool, !jlir<"Array{Float64, 2}">, !jlir.Float64, !jlir.Int64, !jlir.Int64) -> !jlir<"Array{Float64, 2}">
# CHECK-NEXT:     %54 = "jlir.==="(%46, %35) : (!jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%54)[^bb30, ^bb29] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb29:  // pred: ^bb28
# CHECK-NEXT:     %55 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %56 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%55, %56, %47)[^bb31] : (!jlir.Int64, !jlir.Int64, !jlir.Bool) -> ()
# CHECK-NEXT:   ^bb30:  // pred: ^bb28
# CHECK-NEXT:     %57 = "jlir.add_int"(%46, %0) : (!jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %58 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%57, %57, %58)[^bb31] : (!jlir.Int64, !jlir.Int64, !jlir.Bool) -> ()
# CHECK-NEXT:   ^bb31(%59: !jlir.Int64, %60: !jlir.Int64, %61: !jlir.Bool):  // 2 preds: ^bb29, ^bb30
# CHECK-NEXT:     %62 = "jlir.not_int"(%61) : (!jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%62)[^bb33, ^bb32] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb32:  // pred: ^bb31
# CHECK-NEXT:     "jlir.goto"(%59, %60)[^bb28] : (!jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb33:  // 2 preds: ^bb27, ^bb31
# CHECK-NEXT:     %63 = "jlir.==="(%31, %20) : (!jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%63)[^bb35, ^bb34] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb34:  // pred: ^bb33
# CHECK-NEXT:     %64 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %65 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %66 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%64, %65, %66)[^bb36] : (!jlir.Int64, !jlir.Int64, !jlir.Bool) -> ()
# CHECK-NEXT:   ^bb35:  // pred: ^bb33
# CHECK-NEXT:     %67 = "jlir.add_int"(%31, %0) : (!jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %68 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%67, %67, %68)[^bb36] : (!jlir.Int64, !jlir.Int64, !jlir.Bool) -> ()
# CHECK-NEXT:   ^bb36(%69: !jlir.Int64, %70: !jlir.Int64, %71: !jlir.Bool):  // 2 preds: ^bb34, ^bb35
# CHECK-NEXT:     %72 = "jlir.not_int"(%71) : (!jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%72)[^bb38, ^bb37] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb37:  // pred: ^bb36
# CHECK-NEXT:     "jlir.goto"(%69, %70)[^bb19] : (!jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb38:  // 2 preds: ^bb18, ^bb36
# CHECK-NEXT:     %73 = "jlir.==="(%15, %4) : (!jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%73)[^bb40, ^bb39] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb39:  // pred: ^bb38
# CHECK-NEXT:     %74 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %75 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %76 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%74, %75, %76)[^bb41] : (!jlir.Int64, !jlir.Int64, !jlir.Bool) -> ()
# CHECK-NEXT:   ^bb40:  // pred: ^bb38
# CHECK-NEXT:     %77 = "jlir.add_int"(%15, %0) : (!jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %78 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%77, %77, %78)[^bb41] : (!jlir.Int64, !jlir.Int64, !jlir.Bool) -> ()
# CHECK-NEXT:   ^bb41(%79: !jlir.Int64, %80: !jlir.Int64, %81: !jlir.Bool):  // 2 preds: ^bb39, ^bb40
# CHECK-NEXT:     %82 = "jlir.not_int"(%81) : (!jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%82)[^bb43, ^bb42] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb42:  // pred: ^bb41
# CHECK-NEXT:     "jlir.goto"(%79, %80)[^bb10] : (!jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb43:  // 2 preds: ^bb9, ^bb41
# CHECK-NEXT:     %83 = "jlir.constant"() {value = #jlir.nothing} : () -> !jlir.Nothing
# CHECK-NEXT:     "jlir.return"(%83) : (!jlir.Nothing) -> ()
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK: error: lowering to LLVM dialect failed
# CHECK-NEXT: error: module verification failed
