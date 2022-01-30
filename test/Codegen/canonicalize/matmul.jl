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



# CHECK: Core.MethodMatch(Tuple{typeof(Main.Main.matmul!), Matrix{Float64}, Matrix{Float64}, Matrix{Float64}}, svec(), matmul!(C, A, B) in Main.Main at /{{.*}}/test/Codegen/canonicalize/matmul.jl:4, true)after translating to MLIR in JLIR dialect:module  {
# CHECK-NEXT:   func nested @"Tuple{typeof(Main.matmul!), Array{Float64, 2}, Array{Float64, 2}, Array{Float64, 2}}"(%arg0: !jlir<"typeof(Main.matmul!)">, %arg1: !jlir<"Array{Float64, 2}">, %arg2: !jlir<"Array{Float64, 2}">, %arg3: !jlir<"Array{Float64, 2}">) -> !jlir.Nothing attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     "jlir.goto"()[^bb1] : () -> ()
# CHECK-NEXT:   ^bb1:  // pred: ^bb0
# CHECK-NEXT:     %0 = "jlir.constant"() {value = #jlir.Core.arraysize} : () -> !jlir<"typeof(Core.arraysize)">
# CHECK-NEXT:     %1 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %2 = "jlir.call"(%0, %arg2, %1) : (!jlir<"typeof(Core.arraysize)">, !jlir<"Array{Float64, 2}">, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %3 = "jlir.constant"() {value = #jlir<"#<intrinsic #29 sle_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %4 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %5 = "jlir.call"(%3, %4, %2) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     %6 = "jlir.constant"() {value = #jlir.ifelse} : () -> !jlir<"typeof(ifelse)">
# CHECK-NEXT:     %7 = "jlir.constant"() {value = #jlir<"0">} : () -> !jlir.Int64
# CHECK-NEXT:     %8 = "jlir.call"(%6, %5, %2, %7) : (!jlir<"typeof(ifelse)">, !jlir.Bool, !jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %9 = "jlir.constant"() {value = #jlir<"#<intrinsic #27 slt_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %10 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %11 = "jlir.call"(%9, %8, %10) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%11)[^bb3, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb2:  // pred: ^bb1
# CHECK-NEXT:     %12 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     %13 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %14 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%12, %13, %14)[^bb4] : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb3:  // pred: ^bb1
# CHECK-NEXT:     %15 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK-NEXT:     %16 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %17 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%15, %16, %17)[^bb4] : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb4(%18: !jlir.Bool, %19: !jlir.Int64, %20: !jlir.Int64):  // 2 preds: ^bb2, ^bb3
# CHECK-NEXT:     %21 = "jlir.constant"() {value = #jlir<"#<intrinsic #44 not_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %22 = "jlir.call"(%21, %18) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%22, %19, %20)[^bb28, ^bb5] {operand_segment_sizes = dense<[1, 0, 2]> : vector<3xi32>} : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb5(%23: !jlir.Int64, %24: !jlir.Int64):  // 2 preds: ^bb4, ^bb27
# CHECK-NEXT:     %25 = "jlir.constant"() {value = #jlir.Core.arraysize} : () -> !jlir<"typeof(Core.arraysize)">
# CHECK-NEXT:     %26 = "jlir.constant"() {value = #jlir<"2">} : () -> !jlir.Int64
# CHECK-NEXT:     %27 = "jlir.call"(%25, %arg2, %26) : (!jlir<"typeof(Core.arraysize)">, !jlir<"Array{Float64, 2}">, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %28 = "jlir.constant"() {value = #jlir<"#<intrinsic #29 sle_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %29 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %30 = "jlir.call"(%28, %29, %27) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     %31 = "jlir.constant"() {value = #jlir.ifelse} : () -> !jlir<"typeof(ifelse)">
# CHECK-NEXT:     %32 = "jlir.constant"() {value = #jlir<"0">} : () -> !jlir.Int64
# CHECK-NEXT:     %33 = "jlir.call"(%31, %30, %27, %32) : (!jlir<"typeof(ifelse)">, !jlir.Bool, !jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %34 = "jlir.constant"() {value = #jlir<"#<intrinsic #27 slt_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %35 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %36 = "jlir.call"(%34, %33, %35) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%36)[^bb7, ^bb6] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb6:  // pred: ^bb5
# CHECK-NEXT:     %37 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     %38 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %39 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%37, %38, %39)[^bb8] : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb7:  // pred: ^bb5
# CHECK-NEXT:     %40 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK-NEXT:     %41 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %42 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%40, %41, %42)[^bb8] : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb8(%43: !jlir.Bool, %44: !jlir.Int64, %45: !jlir.Int64):  // 2 preds: ^bb6, ^bb7
# CHECK-NEXT:     %46 = "jlir.constant"() {value = #jlir<"#<intrinsic #44 not_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %47 = "jlir.call"(%46, %43) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%47, %44, %45)[^bb23, ^bb9] {operand_segment_sizes = dense<[1, 0, 2]> : vector<3xi32>} : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb9(%48: !jlir.Int64, %49: !jlir.Int64):  // 2 preds: ^bb8, ^bb22
# CHECK-NEXT:     %50 = "jlir.constant"() {value = #jlir.Core.arraysize} : () -> !jlir<"typeof(Core.arraysize)">
# CHECK-NEXT:     %51 = "jlir.constant"() {value = #jlir<"2">} : () -> !jlir.Int64
# CHECK-NEXT:     %52 = "jlir.call"(%50, %arg3, %51) : (!jlir<"typeof(Core.arraysize)">, !jlir<"Array{Float64, 2}">, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %53 = "jlir.constant"() {value = #jlir<"#<intrinsic #29 sle_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %54 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %55 = "jlir.call"(%53, %54, %52) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     %56 = "jlir.constant"() {value = #jlir.ifelse} : () -> !jlir<"typeof(ifelse)">
# CHECK-NEXT:     %57 = "jlir.constant"() {value = #jlir<"0">} : () -> !jlir.Int64
# CHECK-NEXT:     %58 = "jlir.call"(%56, %55, %52, %57) : (!jlir<"typeof(ifelse)">, !jlir.Bool, !jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %59 = "jlir.constant"() {value = #jlir<"#<intrinsic #27 slt_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %60 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %61 = "jlir.call"(%59, %58, %60) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%61)[^bb11, ^bb10] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb10:  // pred: ^bb9
# CHECK-NEXT:     %62 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     %63 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %64 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%62, %63, %64)[^bb12] : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb11:  // pred: ^bb9
# CHECK-NEXT:     %65 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK-NEXT:     %66 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %67 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%65, %66, %67)[^bb12] : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb12(%68: !jlir.Bool, %69: !jlir.Int64, %70: !jlir.Int64):  // 2 preds: ^bb10, ^bb11
# CHECK-NEXT:     %71 = "jlir.constant"() {value = #jlir<"#<intrinsic #44 not_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %72 = "jlir.call"(%71, %68) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%72, %69, %70)[^bb18, ^bb13] {operand_segment_sizes = dense<[1, 0, 2]> : vector<3xi32>} : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb13(%73: !jlir.Int64, %74: !jlir.Int64):  // 2 preds: ^bb12, ^bb17
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
# CHECK-NEXT:     %92 = "jlir.call"(%91, %74, %58) : (!jlir<"typeof(===)">, !jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%92)[^bb15, ^bb14] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb14:  // pred: ^bb13
# CHECK-NEXT:     %93 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %94 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %95 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%93, %94, %95)[^bb16] : (!jlir.Int64, !jlir.Int64, !jlir.Bool) -> ()
# CHECK-NEXT:   ^bb15:  // pred: ^bb13
# CHECK-NEXT:     %96 = "jlir.constant"() {value = #jlir<"#<intrinsic #2 add_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %97 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %98 = "jlir.call"(%96, %74, %97) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %99 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%98, %98, %99)[^bb16] : (!jlir.Int64, !jlir.Int64, !jlir.Bool) -> ()
# CHECK-NEXT:   ^bb16(%100: !jlir.Int64, %101: !jlir.Int64, %102: !jlir.Bool):  // 2 preds: ^bb14, ^bb15
# CHECK-NEXT:     %103 = "jlir.constant"() {value = #jlir<"#<intrinsic #44 not_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %104 = "jlir.call"(%103, %102) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%104)[^bb18, ^bb17] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb17:  // pred: ^bb16
# CHECK-NEXT:     "jlir.goto"(%100, %101)[^bb13] : (!jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb18:  // 2 preds: ^bb12, ^bb16
# CHECK-NEXT:     %105 = "jlir.constant"() {value = #jlir<"===">} : () -> !jlir<"typeof(===)">
# CHECK-NEXT:     %106 = "jlir.call"(%105, %49, %33) : (!jlir<"typeof(===)">, !jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%106)[^bb20, ^bb19] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb19:  // pred: ^bb18
# CHECK-NEXT:     %107 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %108 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %109 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%107, %108, %109)[^bb21] : (!jlir.Int64, !jlir.Int64, !jlir.Bool) -> ()
# CHECK-NEXT:   ^bb20:  // pred: ^bb18
# CHECK-NEXT:     %110 = "jlir.constant"() {value = #jlir<"#<intrinsic #2 add_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %111 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %112 = "jlir.call"(%110, %49, %111) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %113 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%112, %112, %113)[^bb21] : (!jlir.Int64, !jlir.Int64, !jlir.Bool) -> ()
# CHECK-NEXT:   ^bb21(%114: !jlir.Int64, %115: !jlir.Int64, %116: !jlir.Bool):  // 2 preds: ^bb19, ^bb20
# CHECK-NEXT:     %117 = "jlir.constant"() {value = #jlir<"#<intrinsic #44 not_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %118 = "jlir.call"(%117, %116) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%118)[^bb23, ^bb22] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb22:  // pred: ^bb21
# CHECK-NEXT:     "jlir.goto"(%114, %115)[^bb9] : (!jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb23:  // 2 preds: ^bb8, ^bb21
# CHECK-NEXT:     %119 = "jlir.constant"() {value = #jlir<"===">} : () -> !jlir<"typeof(===)">
# CHECK-NEXT:     %120 = "jlir.call"(%119, %24, %8) : (!jlir<"typeof(===)">, !jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%120)[^bb25, ^bb24] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb24:  // pred: ^bb23
# CHECK-NEXT:     %121 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %122 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %123 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%121, %122, %123)[^bb26] : (!jlir.Int64, !jlir.Int64, !jlir.Bool) -> ()
# CHECK-NEXT:   ^bb25:  // pred: ^bb23
# CHECK-NEXT:     %124 = "jlir.constant"() {value = #jlir<"#<intrinsic #2 add_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %125 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %126 = "jlir.call"(%124, %24, %125) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %127 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%126, %126, %127)[^bb26] : (!jlir.Int64, !jlir.Int64, !jlir.Bool) -> ()
# CHECK-NEXT:   ^bb26(%128: !jlir.Int64, %129: !jlir.Int64, %130: !jlir.Bool):  // 2 preds: ^bb24, ^bb25
# CHECK-NEXT:     %131 = "jlir.constant"() {value = #jlir<"#<intrinsic #44 not_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %132 = "jlir.call"(%131, %130) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%132)[^bb28, ^bb27] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb27:  // pred: ^bb26
# CHECK-NEXT:     "jlir.goto"(%128, %129)[^bb5] : (!jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb28:  // 2 preds: ^bb4, ^bb26
# CHECK-NEXT:     %133 = "jlir.constant"() {value = #jlir.nothing} : () -> !jlir.Nothing
# CHECK-NEXT:     "jlir.return"(%133) : (!jlir.Nothing) -> ()
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK:   func nested @"Tuple{typeof(Main.matmul!), Array{Float64, 2}, Array{Float64, 2}, Array{Float64, 2}}"(%arg0: !jlir<"typeof(Main.matmul!)">, %arg1: !jlir<"Array{Float64, 2}">, %arg2: !jlir<"Array{Float64, 2}">, %arg3: !jlir<"Array{Float64, 2}">) -> !jlir.Nothing attributes {llvm.emit_c_interface} {
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
