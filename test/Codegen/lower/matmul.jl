# RUN: julia -e "import Brutus; Brutus.lit(:emit_lowered)" --startup-file=no %s 2>&1 | FileCheck %s

# Intentionally bad code
function matmul!(C, A, B)
    for i in 1:size(A, 1)
        for k in 1:size(A, 2)
            for j in 1:size(B, 2)
                @inbounds C[i, j] += A[i, k] * B[k, j]
            end
        end
    end
    return nothing
end

emit(matmul!, Matrix{Float64}, Matrix{Float64}, Matrix{Float64})




# CHECK: module  {
# CHECK-NEXT:   func nested @"Tuple{typeof(Main.matmul!), Array{Float64, 2}, Array{Float64, 2}, Array{Float64, 2}}"(%arg0: !jlir<"typeof(Main.matmul!)">, %arg1: !jlir<"Array{Float64, 2}">, %arg2: !jlir<"Array{Float64, 2}">, %arg3: !jlir<"Array{Float64, 2}">, %arg4: !jlir<"Union{Nothing, Tuple{Int64, Int64}}">, %arg5: !jlir<"Union{Nothing, Tuple{Int64, Int64}}">, %arg6: !jlir.Int64, %arg7: !jlir<"Union{Nothing, Tuple{Int64, Int64}}">, %arg8: !jlir.Int64, %arg9: !jlir.Float64, %arg10: !jlir.Int64) -> !jlir.Nothing attributes {llvm.emit_c_interface} {
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
# CHECK-NEXT:     %76 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK-NEXT:     %77 = "jlir.call"(%75, %76, %arg1, %23, %73) : (!jlir<"typeof(Core.arrayref)">, !jlir.Bool, !jlir<"Array{Float64, 2}">, !jlir.Int64, !jlir.Int64) -> !jlir.Float64
# CHECK-NEXT:     %78 = "jlir.constant"() {value = #jlir.Core.arrayref} : () -> !jlir<"typeof(Core.arrayref)">
# CHECK-NEXT:     %79 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK-NEXT:     %80 = "jlir.call"(%78, %79, %arg2, %23, %48) : (!jlir<"typeof(Core.arrayref)">, !jlir.Bool, !jlir<"Array{Float64, 2}">, !jlir.Int64, !jlir.Int64) -> !jlir.Float64
# CHECK-NEXT:     %81 = "jlir.constant"() {value = #jlir.Core.arrayref} : () -> !jlir<"typeof(Core.arrayref)">
# CHECK-NEXT:     %82 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK-NEXT:     %83 = "jlir.call"(%81, %82, %arg3, %48, %73) : (!jlir<"typeof(Core.arrayref)">, !jlir.Bool, !jlir<"Array{Float64, 2}">, !jlir.Int64, !jlir.Int64) -> !jlir.Float64
# CHECK-NEXT:     %84 = "jlir.constant"() {value = #jlir<"#<intrinsic #14 mul_float>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %85 = "jlir.call"(%84, %80, %83) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Float64, !jlir.Float64) -> !jlir.Float64
# CHECK-NEXT:     %86 = "jlir.constant"() {value = #jlir<"#<intrinsic #12 add_float>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %87 = "jlir.call"(%86, %77, %85) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Float64, !jlir.Float64) -> !jlir.Float64
# CHECK-NEXT:     %88 = "jlir.constant"() {value = #jlir.Core.arrayset} : () -> !jlir<"typeof(Core.arrayset)">
# CHECK-NEXT:     %89 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
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
# CHECK-NEXT:   func nested @"Tuple{typeof(Main.matmul!), Array{Float64, 2}, Array{Float64, 2}, Array{Float64, 2}}"(%arg0: !jlir<"typeof(Main.matmul!)">, %arg1: memref<?x?xf64>, %arg2: memref<?x?xf64>, %arg3: memref<?x?xf64>, %arg4: !jlir<"Union{Nothing, Tuple{Int64, Int64}}">, %arg5: !jlir<"Union{Nothing, Tuple{Int64, Int64}}">, %arg6: i64, %arg7: !jlir<"Union{Nothing, Tuple{Int64, Int64}}">, %arg8: i64, %arg9: f64, %arg10: i64) -> !jlir.Nothing attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     %c2_i64 = constant 2 : i64
# CHECK-NEXT:     %false = constant false
# CHECK-NEXT:     %true = constant true
# CHECK-NEXT:     %c0_i64 = constant 0 : i64
# CHECK-NEXT:     %c1_i64 = constant 1 : i64
# CHECK-NEXT:     %0 = "jlir.convertstd"(%arg1) : (memref<?x?xf64>) -> !jlir<"Array{Float64, 2}">
# CHECK-NEXT:     %1 = "jlir.convertstd"(%arg2) : (memref<?x?xf64>) -> !jlir<"Array{Float64, 2}">
# CHECK-NEXT:     %2 = "jlir.convertstd"(%arg3) : (memref<?x?xf64>) -> !jlir<"Array{Float64, 2}">
# CHECK-NEXT:     %3 = "jlir.convertstd"(%c1_i64) : (i64) -> !jlir.Int64
# CHECK-NEXT:     %4 = "jlir.arraysize"(%1, %3) : (!jlir<"Array{Float64, 2}">, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %5 = "jlir.convertstd"(%4) : (!jlir.Int64) -> i64
# CHECK-NEXT:     %6 = cmpi sle, %c1_i64, %5 : i64
# CHECK-NEXT:     cond_br %6, ^bb1, ^bb2(%c0_i64 : i64)
# CHECK-NEXT:   ^bb1:  // pred: ^bb0
# CHECK-NEXT:     br ^bb2(%5 : i64)
# CHECK-NEXT:   ^bb2(%7: i64):  // 2 preds: ^bb0, ^bb1
# CHECK-NEXT:     %8 = "jlir.convertstd"(%7) : (i64) -> !jlir.Int64
# CHECK-NEXT:     %9 = cmpi slt, %7, %c1_i64 : i64
# CHECK-NEXT:     cond_br %9, ^bb3, ^bb4(%false, %c1_i64, %c1_i64 : i1, i64, i64)
# CHECK-NEXT:   ^bb3:  // pred: ^bb2
# CHECK-NEXT:     %10 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %11 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %12 = "jlir.convertstd"(%10) : (!jlir.Int64) -> i64
# CHECK-NEXT:     %13 = "jlir.convertstd"(%11) : (!jlir.Int64) -> i64
# CHECK-NEXT:     br ^bb4(%true, %12, %13 : i1, i64, i64)
# CHECK-NEXT:   ^bb4(%14: i1, %15: i64, %16: i64):  // 2 preds: ^bb2, ^bb3
# CHECK-NEXT:     %17 = xor %14, %true : i1
# CHECK-NEXT:     cond_br %17, ^bb5(%15, %16 : i64, i64), ^bb27
# CHECK-NEXT:   ^bb5(%18: i64, %19: i64):  // 2 preds: ^bb4, ^bb26
# CHECK-NEXT:     %20 = "jlir.convertstd"(%18) : (i64) -> !jlir.Int64
# CHECK-NEXT:     %21 = "jlir.convertstd"(%19) : (i64) -> !jlir.Int64
# CHECK-NEXT:     %22 = "jlir.convertstd"(%c2_i64) : (i64) -> !jlir.Int64
# CHECK-NEXT:     %23 = "jlir.arraysize"(%1, %22) : (!jlir<"Array{Float64, 2}">, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %24 = "jlir.convertstd"(%23) : (!jlir.Int64) -> i64
# CHECK-NEXT:     %25 = cmpi sle, %c1_i64, %24 : i64
# CHECK-NEXT:     cond_br %25, ^bb6, ^bb7(%c0_i64 : i64)
# CHECK-NEXT:   ^bb6:  // pred: ^bb5
# CHECK-NEXT:     br ^bb7(%24 : i64)
# CHECK-NEXT:   ^bb7(%26: i64):  // 2 preds: ^bb5, ^bb6
# CHECK-NEXT:     %27 = "jlir.convertstd"(%26) : (i64) -> !jlir.Int64
# CHECK-NEXT:     %28 = cmpi slt, %26, %c1_i64 : i64
# CHECK-NEXT:     cond_br %28, ^bb8, ^bb9(%false, %c1_i64, %c1_i64 : i1, i64, i64)
# CHECK-NEXT:   ^bb8:  // pred: ^bb7
# CHECK-NEXT:     %29 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %30 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %31 = "jlir.convertstd"(%29) : (!jlir.Int64) -> i64
# CHECK-NEXT:     %32 = "jlir.convertstd"(%30) : (!jlir.Int64) -> i64
# CHECK-NEXT:     br ^bb9(%true, %31, %32 : i1, i64, i64)
# CHECK-NEXT:   ^bb9(%33: i1, %34: i64, %35: i64):  // 2 preds: ^bb7, ^bb8
# CHECK-NEXT:     %36 = xor %33, %true : i1
# CHECK-NEXT:     cond_br %36, ^bb10(%34, %35 : i64, i64), ^bb23
# CHECK-NEXT:   ^bb10(%37: i64, %38: i64):  // 2 preds: ^bb9, ^bb22
# CHECK-NEXT:     %39 = "jlir.convertstd"(%37) : (i64) -> !jlir.Int64
# CHECK-NEXT:     %40 = "jlir.convertstd"(%38) : (i64) -> !jlir.Int64
# CHECK-NEXT:     %41 = "jlir.arraysize"(%2, %22) : (!jlir<"Array{Float64, 2}">, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %42 = "jlir.convertstd"(%41) : (!jlir.Int64) -> i64
# CHECK-NEXT:     %43 = cmpi sle, %c1_i64, %42 : i64
# CHECK-NEXT:     cond_br %43, ^bb11, ^bb12(%c0_i64 : i64)
# CHECK-NEXT:   ^bb11:  // pred: ^bb10
# CHECK-NEXT:     br ^bb12(%42 : i64)
# CHECK-NEXT:   ^bb12(%44: i64):  // 2 preds: ^bb10, ^bb11
# CHECK-NEXT:     %45 = "jlir.convertstd"(%44) : (i64) -> !jlir.Int64
# CHECK-NEXT:     %46 = cmpi slt, %44, %c1_i64 : i64
# CHECK-NEXT:     cond_br %46, ^bb13, ^bb14(%false, %c1_i64, %c1_i64 : i1, i64, i64)
# CHECK-NEXT:   ^bb13:  // pred: ^bb12
# CHECK-NEXT:     %47 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %48 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %49 = "jlir.convertstd"(%47) : (!jlir.Int64) -> i64
# CHECK-NEXT:     %50 = "jlir.convertstd"(%48) : (!jlir.Int64) -> i64
# CHECK-NEXT:     br ^bb14(%true, %49, %50 : i1, i64, i64)
# CHECK-NEXT:   ^bb14(%51: i1, %52: i64, %53: i64):  // 2 preds: ^bb12, ^bb13
# CHECK-NEXT:     %54 = xor %51, %true : i1
# CHECK-NEXT:     cond_br %54, ^bb15(%52, %53 : i64, i64), ^bb19
# CHECK-NEXT:   ^bb15(%55: i64, %56: i64):  // 2 preds: ^bb14, ^bb18
# CHECK-NEXT:     %57 = "jlir.convertstd"(%55) : (i64) -> !jlir.Int64
# CHECK-NEXT:     %58 = "jlir.convertstd"(%56) : (i64) -> !jlir.Int64
# CHECK-NEXT:     %59 = "jlir.convertstd"(%false) : (i1) -> !jlir.Bool
# CHECK-NEXT:     %60 = "jlir.arrayref"(%59, %0, %20, %57) : (!jlir.Bool, !jlir<"Array{Float64, 2}">, !jlir.Int64, !jlir.Int64) -> !jlir.Float64
# CHECK-NEXT:     %61 = "jlir.arrayref"(%59, %1, %20, %39) : (!jlir.Bool, !jlir<"Array{Float64, 2}">, !jlir.Int64, !jlir.Int64) -> !jlir.Float64
# CHECK-NEXT:     %62 = "jlir.arrayref"(%59, %2, %39, %57) : (!jlir.Bool, !jlir<"Array{Float64, 2}">, !jlir.Int64, !jlir.Int64) -> !jlir.Float64
# CHECK-NEXT:     %63 = "jlir.convertstd"(%61) : (!jlir.Float64) -> f64
# CHECK-NEXT:     %64 = "jlir.convertstd"(%62) : (!jlir.Float64) -> f64
# CHECK-NEXT:     %65 = mulf %63, %64 : f64
# CHECK-NEXT:     %66 = "jlir.convertstd"(%60) : (!jlir.Float64) -> f64
# CHECK-NEXT:     %67 = addf %66, %65 : f64
# CHECK-NEXT:     %68 = "jlir.convertstd"(%67) : (f64) -> !jlir.Float64
# CHECK-NEXT:     %69 = "jlir.arrayset"(%59, %0, %68, %20, %57) : (!jlir.Bool, !jlir<"Array{Float64, 2}">, !jlir.Float64, !jlir.Int64, !jlir.Int64) -> !jlir<"Array{Float64, 2}">
# CHECK-NEXT:     %70 = "jlir.==="(%58, %45) : (!jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     %71 = "jlir.convertstd"(%70) : (!jlir.Bool) -> i1
# CHECK-NEXT:     cond_br %71, ^bb16, ^bb17
# CHECK-NEXT:   ^bb16:  // pred: ^bb15
# CHECK-NEXT:     %72 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %73 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %74 = "jlir.convertstd"(%72) : (!jlir.Int64) -> i64
# CHECK-NEXT:     %75 = "jlir.convertstd"(%73) : (!jlir.Int64) -> i64
# CHECK-NEXT:     br ^bb18(%74, %75, %true : i64, i64, i1)
# CHECK-NEXT:   ^bb17:  // pred: ^bb15
# CHECK-NEXT:     %76 = addi %56, %c1_i64 : i64
# CHECK-NEXT:     br ^bb18(%76, %76, %false : i64, i64, i1)
# CHECK-NEXT:   ^bb18(%77: i64, %78: i64, %79: i1):  // 2 preds: ^bb16, ^bb17
# CHECK-NEXT:     %80 = xor %79, %true : i1
# CHECK-NEXT:     cond_br %80, ^bb15(%77, %78 : i64, i64), ^bb19
# CHECK-NEXT:   ^bb19:  // 2 preds: ^bb14, ^bb18
# CHECK-NEXT:     %81 = "jlir.==="(%40, %27) : (!jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     %82 = "jlir.convertstd"(%81) : (!jlir.Bool) -> i1
# CHECK-NEXT:     cond_br %82, ^bb20, ^bb21
# CHECK-NEXT:   ^bb20:  // pred: ^bb19
# CHECK-NEXT:     %83 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %84 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %85 = "jlir.convertstd"(%83) : (!jlir.Int64) -> i64
# CHECK-NEXT:     %86 = "jlir.convertstd"(%84) : (!jlir.Int64) -> i64
# CHECK-NEXT:     br ^bb22(%85, %86, %true : i64, i64, i1)
# CHECK-NEXT:   ^bb21:  // pred: ^bb19
# CHECK-NEXT:     %87 = addi %38, %c1_i64 : i64
# CHECK-NEXT:     br ^bb22(%87, %87, %false : i64, i64, i1)
# CHECK-NEXT:   ^bb22(%88: i64, %89: i64, %90: i1):  // 2 preds: ^bb20, ^bb21
# CHECK-NEXT:     %91 = xor %90, %true : i1
# CHECK-NEXT:     cond_br %91, ^bb10(%88, %89 : i64, i64), ^bb23
# CHECK-NEXT:   ^bb23:  // 2 preds: ^bb9, ^bb22
# CHECK-NEXT:     %92 = "jlir.==="(%21, %8) : (!jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     %93 = "jlir.convertstd"(%92) : (!jlir.Bool) -> i1
# CHECK-NEXT:     cond_br %93, ^bb24, ^bb25
# CHECK-NEXT:   ^bb24:  // pred: ^bb23
# CHECK-NEXT:     %94 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %95 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %96 = "jlir.convertstd"(%94) : (!jlir.Int64) -> i64
# CHECK-NEXT:     %97 = "jlir.convertstd"(%95) : (!jlir.Int64) -> i64
# CHECK-NEXT:     br ^bb26(%96, %97, %true : i64, i64, i1)
# CHECK-NEXT:   ^bb25:  // pred: ^bb23
# CHECK-NEXT:     %98 = addi %19, %c1_i64 : i64
# CHECK-NEXT:     br ^bb26(%98, %98, %false : i64, i64, i1)
# CHECK-NEXT:   ^bb26(%99: i64, %100: i64, %101: i1):  // 2 preds: ^bb24, ^bb25
# CHECK-NEXT:     %102 = xor %101, %true : i1
# CHECK-NEXT:     cond_br %102, ^bb5(%99, %100 : i64, i64), ^bb27
# CHECK-NEXT:   ^bb27:  // 2 preds: ^bb4, ^bb26
# CHECK-NEXT:     %103 = "jlir.constant"() {value = #jlir.nothing} : () -> !jlir.Nothing
# CHECK-NEXT:     return %103 : !jlir.Nothing
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK: error: lowering to LLVM dialect failed
# CHECK-NEXT: error: module verification failed

# CHECK: module  {
# CHECK-NEXT:   llvm.func @"Tuple{typeof(Main.matmul!), Array{Float64, 2}, Array{Float64, 2}, Array{Float64, 2}}"(%arg0: !llvm.ptr<struct<"struct_jl_value_type", opaque>>, %arg1: !llvm.ptr<f64>, %arg2: !llvm.ptr<f64>, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: !llvm.ptr<f64>, %arg9: !llvm.ptr<f64>, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: !llvm.ptr<f64>, %arg16: !llvm.ptr<f64>, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64, %arg21: i64, %arg22: !llvm.ptr<struct<"struct_jl_value_type", opaque>>, %arg23: !llvm.ptr<struct<"struct_jl_value_type", opaque>>, %arg24: i64, %arg25: !llvm.ptr<struct<"struct_jl_value_type", opaque>>, %arg26: i64, %arg27: f64, %arg28: i64) -> !llvm.ptr<struct<"struct_jl_value_type", opaque>> attributes {llvm.emit_c_interface, sym_visibility = "nested"} {
# CHECK-NEXT:     %0 = llvm.mlir.undef : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %1 = llvm.insertvalue %arg1, %0[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %2 = llvm.insertvalue %arg2, %1[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %3 = llvm.insertvalue %arg3, %2[2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %4 = llvm.insertvalue %arg4, %3[3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %5 = llvm.insertvalue %arg6, %4[4, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %6 = llvm.insertvalue %arg5, %5[3, 1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %7 = llvm.insertvalue %arg7, %6[4, 1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %8 = llvm.mlir.undef : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %9 = llvm.insertvalue %arg8, %8[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %10 = llvm.insertvalue %arg9, %9[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %11 = llvm.insertvalue %arg10, %10[2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %12 = llvm.insertvalue %arg11, %11[3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %13 = llvm.insertvalue %arg13, %12[4, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %14 = llvm.insertvalue %arg12, %13[3, 1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %15 = llvm.insertvalue %arg14, %14[4, 1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %16 = llvm.mlir.undef : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %17 = llvm.insertvalue %arg15, %16[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %18 = llvm.insertvalue %arg16, %17[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %19 = llvm.insertvalue %arg17, %18[2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %20 = llvm.insertvalue %arg18, %19[3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %21 = llvm.insertvalue %arg20, %20[4, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %22 = llvm.insertvalue %arg19, %21[3, 1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %23 = llvm.insertvalue %arg21, %22[4, 1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %24 = llvm.mlir.constant({{[0-9]+}} : i64) : i64
# CHECK-NEXT:     %25 = llvm.mlir.constant(false) : i1
# CHECK-NEXT:     %26 = llvm.mlir.constant(true) : i1
# CHECK-NEXT:     %27 = llvm.mlir.constant({{[0-9]+}} : i64) : i64
# CHECK-NEXT:     %28 = llvm.mlir.constant({{[0-9]+}} : i64) : i64
# CHECK-NEXT:     %29 = "jlir.arraysize"(%15, %28) : (!llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>, i64) -> !jlir.Int64
# CHECK-NEXT:     %30 = llvm.icmp "sle" %28, %29 : i64
# CHECK-NEXT:     llvm.cond_br %30, ^bb1, ^bb2(%27 : i64)
# CHECK-NEXT:   ^bb1:  // pred: ^bb0
# CHECK-NEXT:     llvm.br ^bb2(%29 : !jlir.Int64)
# CHECK-NEXT:   ^bb2(%31: i64):  // 2 preds: ^bb0, ^bb1
# CHECK-NEXT:     %32 = llvm.icmp "slt" %31, %28 : i64
# CHECK-NEXT:     llvm.cond_br %32, ^bb3, ^bb4(%25, %28, %28 : i1, i64, i64)
# CHECK-NEXT:   ^bb3:  // pred: ^bb2
# CHECK-NEXT:     %33 = llvm.mlir.undef : i64
# CHECK-NEXT:     %34 = llvm.mlir.undef : i64
# CHECK-NEXT:     llvm.br ^bb4(%26, %33, %34 : i1, i64, i64)
# CHECK-NEXT:   ^bb4(%35: i1, %36: i64, %37: i64):  // 2 preds: ^bb2, ^bb3
# CHECK-NEXT:     %38 = llvm.xor %35, %26  : i1
# CHECK-NEXT:     llvm.cond_br %38, ^bb5(%36, %37 : i64, i64), ^bb27
# CHECK-NEXT:   ^bb5(%39: i64, %40: i64):  // 2 preds: ^bb4, ^bb26
# CHECK-NEXT:     %41 = "jlir.arraysize"(%15, %24) : (!llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>, i64) -> !jlir.Int64
# CHECK-NEXT:     %42 = llvm.icmp "sle" %28, %41 : i64
# CHECK-NEXT:     llvm.cond_br %42, ^bb6, ^bb7(%27 : i64)
# CHECK-NEXT:   ^bb6:  // pred: ^bb5
# CHECK-NEXT:     llvm.br ^bb7(%41 : !jlir.Int64)
# CHECK-NEXT:   ^bb7(%43: i64):  // 2 preds: ^bb5, ^bb6
# CHECK-NEXT:     %44 = llvm.icmp "slt" %43, %28 : i64
# CHECK-NEXT:     llvm.cond_br %44, ^bb8, ^bb9(%25, %28, %28 : i1, i64, i64)
# CHECK-NEXT:   ^bb8:  // pred: ^bb7
# CHECK-NEXT:     %45 = llvm.mlir.undef : i64
# CHECK-NEXT:     %46 = llvm.mlir.undef : i64
# CHECK-NEXT:     llvm.br ^bb9(%26, %45, %46 : i1, i64, i64)
# CHECK-NEXT:   ^bb9(%47: i1, %48: i64, %49: i64):  // 2 preds: ^bb7, ^bb8
# CHECK-NEXT:     %50 = llvm.xor %47, %26  : i1
# CHECK-NEXT:     llvm.cond_br %50, ^bb10(%48, %49 : i64, i64), ^bb23
# CHECK-NEXT:   ^bb10(%51: i64, %52: i64):  // 2 preds: ^bb9, ^bb22
# CHECK-NEXT:     %53 = "jlir.arraysize"(%23, %24) : (!llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>, i64) -> !jlir.Int64
# CHECK-NEXT:     %54 = llvm.icmp "sle" %28, %53 : i64
# CHECK-NEXT:     llvm.cond_br %54, ^bb11, ^bb12(%27 : i64)
# CHECK-NEXT:   ^bb11:  // pred: ^bb10
# CHECK-NEXT:     llvm.br ^bb12(%53 : !jlir.Int64)
# CHECK-NEXT:   ^bb12(%55: i64):  // 2 preds: ^bb10, ^bb11
# CHECK-NEXT:     %56 = llvm.icmp "slt" %55, %28 : i64
# CHECK-NEXT:     llvm.cond_br %56, ^bb13, ^bb14(%25, %28, %28 : i1, i64, i64)
# CHECK-NEXT:   ^bb13:  // pred: ^bb12
# CHECK-NEXT:     %57 = llvm.mlir.undef : i64
# CHECK-NEXT:     %58 = llvm.mlir.undef : i64
# CHECK-NEXT:     llvm.br ^bb14(%26, %57, %58 : i1, i64, i64)
# CHECK-NEXT:   ^bb14(%59: i1, %60: i64, %61: i64):  // 2 preds: ^bb12, ^bb13
# CHECK-NEXT:     %62 = llvm.xor %59, %26  : i1
# CHECK-NEXT:     llvm.cond_br %62, ^bb15(%60, %61 : i64, i64), ^bb19
# CHECK-NEXT:   ^bb15(%63: i64, %64: i64):  // 2 preds: ^bb14, ^bb18
# CHECK-NEXT:     %65 = "jlir.arrayref"(%25, %7, %39, %63) : (i1, !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>, i64, i64) -> !jlir.Float64
# CHECK-NEXT:     %66 = "jlir.arrayref"(%25, %15, %39, %51) : (i1, !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>, i64, i64) -> !jlir.Float64
# CHECK-NEXT:     %67 = "jlir.arrayref"(%25, %23, %51, %63) : (i1, !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>, i64, i64) -> !jlir.Float64
# CHECK-NEXT:     %68 = llvm.fmul %66, %67  : f64
# CHECK-NEXT:     %69 = llvm.fadd %65, %68  : f64
# CHECK-NEXT:     %70 = "jlir.arrayset"(%25, %7, %69, %39, %63) : (i1, !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>, f64, i64, i64) -> !jlir<"Array{Float64, 2}">
# CHECK-NEXT:     %71 = llvm.icmp "eq" %64, %55 : i64
# CHECK-NEXT:     llvm.cond_br %71, ^bb16, ^bb17
# CHECK-NEXT:   ^bb16:  // pred: ^bb15
# CHECK-NEXT:     %72 = llvm.mlir.undef : i64
# CHECK-NEXT:     %73 = llvm.mlir.undef : i64
# CHECK-NEXT:     llvm.br ^bb18(%72, %73, %26 : i64, i64, i1)
# CHECK-NEXT:   ^bb17:  // pred: ^bb15
# CHECK-NEXT:     %74 = llvm.add %64, %28  : i64
# CHECK-NEXT:     llvm.br ^bb18(%74, %74, %25 : i64, i64, i1)
# CHECK-NEXT:   ^bb18(%75: i64, %76: i64, %77: i1):  // 2 preds: ^bb16, ^bb17
# CHECK-NEXT:     %78 = llvm.xor %77, %26  : i1
# CHECK-NEXT:     llvm.cond_br %78, ^bb15(%75, %76 : i64, i64), ^bb19
# CHECK-NEXT:   ^bb19:  // 2 preds: ^bb14, ^bb18
# CHECK-NEXT:     %79 = llvm.icmp "eq" %52, %43 : i64
# CHECK-NEXT:     llvm.cond_br %79, ^bb20, ^bb21
# CHECK-NEXT:   ^bb20:  // pred: ^bb19
# CHECK-NEXT:     %80 = llvm.mlir.undef : i64
# CHECK-NEXT:     %81 = llvm.mlir.undef : i64
# CHECK-NEXT:     llvm.br ^bb22(%80, %81, %26 : i64, i64, i1)
# CHECK-NEXT:   ^bb21:  // pred: ^bb19
# CHECK-NEXT:     %82 = llvm.add %52, %28  : i64
# CHECK-NEXT:     llvm.br ^bb22(%82, %82, %25 : i64, i64, i1)
# CHECK-NEXT:   ^bb22(%83: i64, %84: i64, %85: i1):  // 2 preds: ^bb20, ^bb21
# CHECK-NEXT:     %86 = llvm.xor %85, %26  : i1
# CHECK-NEXT:     llvm.cond_br %86, ^bb10(%83, %84 : i64, i64), ^bb23
# CHECK-NEXT:   ^bb23:  // 2 preds: ^bb9, ^bb22
# CHECK-NEXT:     %87 = llvm.icmp "eq" %40, %31 : i64
# CHECK-NEXT:     llvm.cond_br %87, ^bb24, ^bb25
# CHECK-NEXT:   ^bb24:  // pred: ^bb23
# CHECK-NEXT:     %88 = llvm.mlir.undef : i64
# CHECK-NEXT:     %89 = llvm.mlir.undef : i64
# CHECK-NEXT:     llvm.br ^bb26(%88, %89, %26 : i64, i64, i1)
# CHECK-NEXT:   ^bb25:  // pred: ^bb23
# CHECK-NEXT:     %90 = llvm.add %40, %28  : i64
# CHECK-NEXT:     llvm.br ^bb26(%90, %90, %25 : i64, i64, i1)
# CHECK-NEXT:   ^bb26(%91: i64, %92: i64, %93: i1):  // 2 preds: ^bb24, ^bb25
# CHECK-NEXT:     %94 = llvm.xor %93, %26  : i1
# CHECK-NEXT:     llvm.cond_br %94, ^bb5(%91, %92 : i64, i64), ^bb27
# CHECK-NEXT:   ^bb27:  // 2 preds: ^bb4, ^bb26
# CHECK-NEXT:     %95 = llvm.mlir.constant({{[0-9]+}} : i64) : i64
# CHECK-NEXT:     %96 = llvm.inttoptr %95 : i64 to !llvm.ptr<struct<"struct_jl_value_type", opaque>>
# CHECK-NEXT:     llvm.return %96 : !llvm.ptr<struct<"struct_jl_value_type", opaque>>
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @"_mlir_ciface_Tuple{typeof(Main.matmul!), Array{Float64, 2}, Array{Float64, 2}, Array{Float64, 2}}"(%arg0: !llvm.ptr<struct<"struct_jl_value_type", opaque>>, %arg1: !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>>, %arg2: !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>>, %arg3: !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>>, %arg4: !llvm.ptr<struct<"struct_jl_value_type", opaque>>, %arg5: !llvm.ptr<struct<"struct_jl_value_type", opaque>>, %arg6: i64, %arg7: !llvm.ptr<struct<"struct_jl_value_type", opaque>>, %arg8: i64, %arg9: f64, %arg10: i64) -> !llvm.ptr<struct<"struct_jl_value_type", opaque>> attributes {llvm.emit_c_interface, sym_visibility = "nested"} {
# CHECK-NEXT:     %0 = llvm.load %arg1 : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>>
# CHECK-NEXT:     %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %5 = llvm.extractvalue %0[3, 1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %6 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %7 = llvm.extractvalue %0[4, 1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %8 = llvm.load %arg2 : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>>
# CHECK-NEXT:     %9 = llvm.extractvalue %8[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %10 = llvm.extractvalue %8[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %11 = llvm.extractvalue %8[2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %12 = llvm.extractvalue %8[3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %13 = llvm.extractvalue %8[3, 1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %14 = llvm.extractvalue %8[4, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %15 = llvm.extractvalue %8[4, 1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %16 = llvm.load %arg3 : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>>
# CHECK-NEXT:     %17 = llvm.extractvalue %16[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %18 = llvm.extractvalue %16[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %19 = llvm.extractvalue %16[2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %20 = llvm.extractvalue %16[3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %21 = llvm.extractvalue %16[3, 1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %22 = llvm.extractvalue %16[4, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %23 = llvm.extractvalue %16[4, 1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %24 = llvm.call @"Tuple{typeof(Main.matmul!), Array{Float64, 2}, Array{Float64, 2}, Array{Float64, 2}}"(%arg0, %1, %2, %3, %4, %5, %6, %7, %9, %10, %11, %12, %13, %14, %15, %17, %18, %19, %20, %21, %22, %23, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10) : (!llvm.ptr<struct<"struct_jl_value_type", opaque>>, !llvm.ptr<f64>, !llvm.ptr<f64>, i64, i64, i64, i64, i64, !llvm.ptr<f64>, !llvm.ptr<f64>, i64, i64, i64, i64, i64, !llvm.ptr<f64>, !llvm.ptr<f64>, i64, i64, i64, i64, i64, !llvm.ptr<struct<"struct_jl_value_type", opaque>>, !llvm.ptr<struct<"struct_jl_value_type", opaque>>, i64, !llvm.ptr<struct<"struct_jl_value_type", opaque>>, i64, f64, i64) -> !llvm.ptr<struct<"struct_jl_value_type", opaque>>
# CHECK-NEXT:     llvm.return %24 : !llvm.ptr<struct<"struct_jl_value_type", opaque>>
# CHECK-NEXT:   }
# CHECK-NEXT: }
