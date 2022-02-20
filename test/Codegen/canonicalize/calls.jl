# RUN: julia -e "import Brutus; Brutus.lit(:emit_optimized)" --startup-file=no %s 2>&1 | FileCheck %s

function calls()
    f = rand(Bool) ? (+) : (-)
    return f(1, 1)
end
emit(calls)





# CHECK: module  {
# CHECK-NEXT:   func nested @"Tuple{typeof(Main.calls)}"(%arg0: !jlir<"typeof(Main.calls)">, %arg1: !jlir<"Union{typeof(Base.:(+)), typeof(Base.:(-))}">, %arg2: !jlir<"Union{typeof(Base.:(+)), typeof(Base.:(-))}">) -> !jlir.Int64 attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     "jlir.goto"()[^bb1] : () -> ()
# CHECK-NEXT:   ^bb1:  // pred: ^bb0
# CHECK-NEXT:     %0 = "jlir.unimplemented"() : () -> !jlir.Task
# CHECK-NEXT:     %1 = "jlir.constant"() {value = #jlir.getfield} : () -> !jlir<"typeof(getfield)">
# CHECK-NEXT:     %2 = "jlir.constant"() {value = #jlir<":rngState0">} : () -> !jlir.Symbol
# CHECK-NEXT:     %3 = "jlir.call"(%1, %0, %2) : (!jlir<"typeof(getfield)">, !jlir.Task, !jlir.Symbol) -> !jlir.UInt64
# CHECK-NEXT:     %4 = "jlir.constant"() {value = #jlir.getfield} : () -> !jlir<"typeof(getfield)">
# CHECK-NEXT:     %5 = "jlir.constant"() {value = #jlir<":rngState1">} : () -> !jlir.Symbol
# CHECK-NEXT:     %6 = "jlir.call"(%4, %0, %5) : (!jlir<"typeof(getfield)">, !jlir.Task, !jlir.Symbol) -> !jlir.UInt64
# CHECK-NEXT:     %7 = "jlir.constant"() {value = #jlir.getfield} : () -> !jlir<"typeof(getfield)">
# CHECK-NEXT:     %8 = "jlir.constant"() {value = #jlir<":rngState2">} : () -> !jlir.Symbol
# CHECK-NEXT:     %9 = "jlir.call"(%7, %0, %8) : (!jlir<"typeof(getfield)">, !jlir.Task, !jlir.Symbol) -> !jlir.UInt64
# CHECK-NEXT:     %10 = "jlir.constant"() {value = #jlir.getfield} : () -> !jlir<"typeof(getfield)">
# CHECK-NEXT:     %11 = "jlir.constant"() {value = #jlir<":rngState3">} : () -> !jlir.Symbol
# CHECK-NEXT:     %12 = "jlir.call"(%10, %0, %11) : (!jlir<"typeof(getfield)">, !jlir.Task, !jlir.Symbol) -> !jlir.UInt64
# CHECK-NEXT:     %13 = "jlir.constant"() {value = #jlir<"#<intrinsic #2 add_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %14 = "jlir.call"(%13, %3, %12) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %15 = "jlir.constant"() {value = #jlir<"#<intrinsic #44 shl_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %16 = "jlir.constant"() {value = #jlir<"0x0000000000000017">} : () -> !jlir.UInt64
# CHECK-NEXT:     %17 = "jlir.call"(%15, %14, %16) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %18 = "jlir.constant"() {value = #jlir<"#<intrinsic #45 lshr_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %19 = "jlir.constant"() {value = #jlir<"0xffffffffffffffe9">} : () -> !jlir.UInt64
# CHECK-NEXT:     %20 = "jlir.call"(%18, %14, %19) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %21 = "jlir.constant"() {value = #jlir.Core.ifelse} : () -> !jlir<"typeof(Core.ifelse)">
# CHECK-NEXT:     %22 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     %23 = "jlir.call"(%21, %22, %17, %20) : (!jlir<"typeof(Core.ifelse)">, !jlir.Bool, !jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %24 = "jlir.constant"() {value = #jlir<"#<intrinsic #45 lshr_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %25 = "jlir.constant"() {value = #jlir<"0x0000000000000029">} : () -> !jlir.UInt64
# CHECK-NEXT:     %26 = "jlir.call"(%24, %14, %25) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %27 = "jlir.constant"() {value = #jlir<"#<intrinsic #44 shl_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %28 = "jlir.constant"() {value = #jlir<"0xffffffffffffffd7">} : () -> !jlir.UInt64
# CHECK-NEXT:     %29 = "jlir.call"(%27, %14, %28) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %30 = "jlir.constant"() {value = #jlir.Core.ifelse} : () -> !jlir<"typeof(Core.ifelse)">
# CHECK-NEXT:     %31 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     %32 = "jlir.call"(%30, %31, %26, %29) : (!jlir<"typeof(Core.ifelse)">, !jlir.Bool, !jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %33 = "jlir.constant"() {value = #jlir<"#<intrinsic #41 or_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %34 = "jlir.call"(%33, %23, %32) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %35 = "jlir.constant"() {value = #jlir<"#<intrinsic #2 add_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %36 = "jlir.call"(%35, %34, %3) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %37 = "jlir.constant"() {value = #jlir<"#<intrinsic #44 shl_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %38 = "jlir.constant"() {value = #jlir<"0x0000000000000011">} : () -> !jlir.UInt64
# CHECK-NEXT:     %39 = "jlir.call"(%37, %6, %38) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %40 = "jlir.constant"() {value = #jlir<"#<intrinsic #45 lshr_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %41 = "jlir.constant"() {value = #jlir<"0xffffffffffffffef">} : () -> !jlir.UInt64
# CHECK-NEXT:     %42 = "jlir.call"(%40, %6, %41) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %43 = "jlir.constant"() {value = #jlir.Core.ifelse} : () -> !jlir<"typeof(Core.ifelse)">
# CHECK-NEXT:     %44 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     %45 = "jlir.call"(%43, %44, %39, %42) : (!jlir<"typeof(Core.ifelse)">, !jlir.Bool, !jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %46 = "jlir.constant"() {value = #jlir<"#<intrinsic #42 xor_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %47 = "jlir.call"(%46, %9, %3) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %48 = "jlir.constant"() {value = #jlir<"#<intrinsic #42 xor_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %49 = "jlir.call"(%48, %12, %6) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %50 = "jlir.constant"() {value = #jlir<"#<intrinsic #42 xor_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %51 = "jlir.call"(%50, %6, %47) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %52 = "jlir.constant"() {value = #jlir<"#<intrinsic #42 xor_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %53 = "jlir.call"(%52, %3, %49) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %54 = "jlir.constant"() {value = #jlir<"#<intrinsic #42 xor_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %55 = "jlir.call"(%54, %47, %45) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %56 = "jlir.constant"() {value = #jlir<"#<intrinsic #44 shl_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %57 = "jlir.constant"() {value = #jlir<"0x000000000000002d">} : () -> !jlir.UInt64
# CHECK-NEXT:     %58 = "jlir.call"(%56, %49, %57) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %59 = "jlir.constant"() {value = #jlir<"#<intrinsic #45 lshr_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %60 = "jlir.constant"() {value = #jlir<"0xffffffffffffffd3">} : () -> !jlir.UInt64
# CHECK-NEXT:     %61 = "jlir.call"(%59, %49, %60) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %62 = "jlir.constant"() {value = #jlir.Core.ifelse} : () -> !jlir<"typeof(Core.ifelse)">
# CHECK-NEXT:     %63 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     %64 = "jlir.call"(%62, %63, %58, %61) : (!jlir<"typeof(Core.ifelse)">, !jlir.Bool, !jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %65 = "jlir.constant"() {value = #jlir<"#<intrinsic #45 lshr_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %66 = "jlir.constant"() {value = #jlir<"0x0000000000000013">} : () -> !jlir.UInt64
# CHECK-NEXT:     %67 = "jlir.call"(%65, %49, %66) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %68 = "jlir.constant"() {value = #jlir<"#<intrinsic #44 shl_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %69 = "jlir.constant"() {value = #jlir<"0xffffffffffffffed">} : () -> !jlir.UInt64
# CHECK-NEXT:     %70 = "jlir.call"(%68, %49, %69) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %71 = "jlir.constant"() {value = #jlir.Core.ifelse} : () -> !jlir<"typeof(Core.ifelse)">
# CHECK-NEXT:     %72 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     %73 = "jlir.call"(%71, %72, %67, %70) : (!jlir<"typeof(Core.ifelse)">, !jlir.Bool, !jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %74 = "jlir.constant"() {value = #jlir<"#<intrinsic #41 or_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %75 = "jlir.call"(%74, %64, %73) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %76 = "jlir.constant"() {value = #jlir<"setfield!">} : () -> !jlir<"typeof(setfield!)">
# CHECK-NEXT:     %77 = "jlir.constant"() {value = #jlir<":rngState0">} : () -> !jlir.Symbol
# CHECK-NEXT:     %78 = "jlir.call"(%76, %0, %77, %53) : (!jlir<"typeof(setfield!)">, !jlir.Task, !jlir.Symbol, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %79 = "jlir.constant"() {value = #jlir<"setfield!">} : () -> !jlir<"typeof(setfield!)">
# CHECK-NEXT:     %80 = "jlir.constant"() {value = #jlir<":rngState1">} : () -> !jlir.Symbol
# CHECK-NEXT:     %81 = "jlir.call"(%79, %0, %80, %51) : (!jlir<"typeof(setfield!)">, !jlir.Task, !jlir.Symbol, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %82 = "jlir.constant"() {value = #jlir<"setfield!">} : () -> !jlir<"typeof(setfield!)">
# CHECK-NEXT:     %83 = "jlir.constant"() {value = #jlir<":rngState2">} : () -> !jlir.Symbol
# CHECK-NEXT:     %84 = "jlir.call"(%82, %0, %83, %55) : (!jlir<"typeof(setfield!)">, !jlir.Task, !jlir.Symbol, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %85 = "jlir.constant"() {value = #jlir<"setfield!">} : () -> !jlir<"typeof(setfield!)">
# CHECK-NEXT:     %86 = "jlir.constant"() {value = #jlir<":rngState3">} : () -> !jlir.Symbol
# CHECK-NEXT:     %87 = "jlir.call"(%85, %0, %86, %75) : (!jlir<"typeof(setfield!)">, !jlir.Task, !jlir.Symbol, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %88 = "jlir.constant"() {value = #jlir<"#<intrinsic #45 lshr_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %89 = "jlir.constant"() {value = #jlir<"0x0000000000000038">} : () -> !jlir.UInt64
# CHECK-NEXT:     %90 = "jlir.call"(%88, %36, %89) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %91 = "jlir.constant"() {value = #jlir<"#<intrinsic #44 shl_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %92 = "jlir.constant"() {value = #jlir<"0xffffffffffffffc8">} : () -> !jlir.UInt64
# CHECK-NEXT:     %93 = "jlir.call"(%91, %36, %92) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %94 = "jlir.constant"() {value = #jlir.Core.ifelse} : () -> !jlir<"typeof(Core.ifelse)">
# CHECK-NEXT:     %95 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     %96 = "jlir.call"(%94, %95, %90, %93) : (!jlir<"typeof(Core.ifelse)">, !jlir.Bool, !jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %97 = "jlir.constant"() {value = #jlir<"#<intrinsic #40 and_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %98 = "jlir.constant"() {value = #jlir<"0x0000000000000001">} : () -> !jlir.UInt64
# CHECK-NEXT:     %99 = "jlir.call"(%97, %96, %98) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %100 = "jlir.constant"() {value = #jlir<"===">} : () -> !jlir<"typeof(===)">
# CHECK-NEXT:     %101 = "jlir.constant"() {value = #jlir<"0x0000000000000000">} : () -> !jlir.UInt64
# CHECK-NEXT:     %102 = "jlir.call"(%100, %99, %101) : (!jlir<"typeof(===)">, !jlir.UInt64, !jlir.UInt64) -> !jlir.Bool
# CHECK-NEXT:     %103 = "jlir.constant"() {value = #jlir<"#<intrinsic #40 and_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %104 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     %105 = "jlir.call"(%103, %104, %102) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Bool, !jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     %106 = "jlir.constant"() {value = #jlir<"#<intrinsic #43 not_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %107 = "jlir.call"(%106, %105) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%107)[^bb3, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb2:  // pred: ^bb1
# CHECK-NEXT:     %108 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     %109 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%108, %109)[^bb4] : (!jlir.Bool, !jlir.Bool) -> ()
# CHECK-NEXT:   ^bb3:  // pred: ^bb1
# CHECK-NEXT:     %110 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK-NEXT:     %111 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%110, %111)[^bb4] : (!jlir.Bool, !jlir.Bool) -> ()
# CHECK-NEXT:   ^bb4(%112: !jlir.Bool, %113: !jlir.Bool):  // 2 preds: ^bb2, ^bb3
# CHECK-NEXT:     "jlir.gotoifnot"(%112)[^bb6, ^bb5] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb5:  // pred: ^bb4
# CHECK-NEXT:     %114 = "jlir.constant"() {value = #jlir<"2">} : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%114)[^bb9] : (!jlir.Int64) -> ()
# CHECK-NEXT:   ^bb6:  // pred: ^bb4
# CHECK-NEXT:     "jlir.gotoifnot"(%113)[^bb8, ^bb7] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb7:  // pred: ^bb6
# CHECK-NEXT:     %115 = "jlir.constant"() {value = #jlir<"0">} : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%115)[^bb9] : (!jlir.Int64) -> ()
# CHECK-NEXT:   ^bb8:  // pred: ^bb6
# CHECK-NEXT:     %116 = "jlir.constant"() {value = #jlir.throw} : () -> !jlir<"typeof(throw)">
# CHECK-NEXT:     %117 = "jlir.constant"() {value = #jlir<"ErrorException(\22fatal error in type inference (type bound)\22)">} : () -> !jlir.ErrorException
# CHECK-NEXT:     %118 = "jlir.call"(%116, %117) : (!jlir<"typeof(throw)">, !jlir.ErrorException) -> !jlir<"Union{}">
# CHECK-NEXT:     %119 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.return"(%119) : (!jlir.Int64) -> ()
# CHECK-NEXT:   ^bb9(%120: !jlir.Int64):  // 2 preds: ^bb5, ^bb7
# CHECK-NEXT:     "jlir.return"(%120) : (!jlir.Int64) -> ()
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK: module  {
# CHECK-NEXT:   func nested @"Tuple{typeof(Main.calls)}"(%arg0: !jlir<"typeof(Main.calls)">, %arg1: !jlir<"Union{typeof(Base.:(+)), typeof(Base.:(-))}">, %arg2: !jlir<"Union{typeof(Base.:(+)), typeof(Base.:(-))}">) -> !jlir.Int64 attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     "jlir.goto"()[^bb1] : () -> ()
# CHECK-NEXT:   ^bb1:  // pred: ^bb0
# CHECK-NEXT:     %0 = "jlir.unimplemented"() : () -> !jlir.Task
# CHECK-NEXT:     %1 = "jlir.constant"() {value = #jlir<":rngState0">} : () -> !jlir.Symbol
# CHECK-NEXT:     %2 = "jlir.getfield"(%0, %1) : (!jlir.Task, !jlir.Symbol) -> !jlir.UInt64
# CHECK-NEXT:     %3 = "jlir.constant"() {value = #jlir<":rngState1">} : () -> !jlir.Symbol
# CHECK-NEXT:     %4 = "jlir.getfield"(%0, %3) : (!jlir.Task, !jlir.Symbol) -> !jlir.UInt64
# CHECK-NEXT:     %5 = "jlir.constant"() {value = #jlir<":rngState2">} : () -> !jlir.Symbol
# CHECK-NEXT:     %6 = "jlir.getfield"(%0, %5) : (!jlir.Task, !jlir.Symbol) -> !jlir.UInt64
# CHECK-NEXT:     %7 = "jlir.constant"() {value = #jlir<":rngState3">} : () -> !jlir.Symbol
# CHECK-NEXT:     %8 = "jlir.getfield"(%0, %7) : (!jlir.Task, !jlir.Symbol) -> !jlir.UInt64
# CHECK-NEXT:     %9 = "jlir.add_int"(%2, %8) : (!jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %10 = "jlir.constant"() {value = #jlir<"0x0000000000000017">} : () -> !jlir.UInt64
# CHECK-NEXT:     %11 = "jlir.shl_int"(%9, %10) : (!jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %12 = "jlir.constant"() {value = #jlir<"0xffffffffffffffe9">} : () -> !jlir.UInt64
# CHECK-NEXT:     %13 = "jlir.lshr_int"(%9, %12) : (!jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %14 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     %15 = "jlir.ifelse"(%14, %11, %13) : (!jlir.Bool, !jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %16 = "jlir.constant"() {value = #jlir<"0x0000000000000029">} : () -> !jlir.UInt64
# CHECK-NEXT:     %17 = "jlir.lshr_int"(%9, %16) : (!jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %18 = "jlir.constant"() {value = #jlir<"0xffffffffffffffd7">} : () -> !jlir.UInt64
# CHECK-NEXT:     %19 = "jlir.shl_int"(%9, %18) : (!jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %20 = "jlir.ifelse"(%14, %17, %19) : (!jlir.Bool, !jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %21 = "jlir.or_int"(%15, %20) : (!jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %22 = "jlir.add_int"(%21, %2) : (!jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %23 = "jlir.constant"() {value = #jlir<"0x0000000000000011">} : () -> !jlir.UInt64
# CHECK-NEXT:     %24 = "jlir.shl_int"(%4, %23) : (!jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %25 = "jlir.constant"() {value = #jlir<"0xffffffffffffffef">} : () -> !jlir.UInt64
# CHECK-NEXT:     %26 = "jlir.lshr_int"(%4, %25) : (!jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %27 = "jlir.ifelse"(%14, %24, %26) : (!jlir.Bool, !jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %28 = "jlir.xor_int"(%6, %2) : (!jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %29 = "jlir.xor_int"(%8, %4) : (!jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %30 = "jlir.xor_int"(%4, %28) : (!jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %31 = "jlir.xor_int"(%2, %29) : (!jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %32 = "jlir.xor_int"(%28, %27) : (!jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %33 = "jlir.constant"() {value = #jlir<"0x000000000000002d">} : () -> !jlir.UInt64
# CHECK-NEXT:     %34 = "jlir.shl_int"(%29, %33) : (!jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %35 = "jlir.constant"() {value = #jlir<"0xffffffffffffffd3">} : () -> !jlir.UInt64
# CHECK-NEXT:     %36 = "jlir.lshr_int"(%29, %35) : (!jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %37 = "jlir.ifelse"(%14, %34, %36) : (!jlir.Bool, !jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %38 = "jlir.constant"() {value = #jlir<"0x0000000000000013">} : () -> !jlir.UInt64
# CHECK-NEXT:     %39 = "jlir.lshr_int"(%29, %38) : (!jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %40 = "jlir.constant"() {value = #jlir<"0xffffffffffffffed">} : () -> !jlir.UInt64
# CHECK-NEXT:     %41 = "jlir.shl_int"(%29, %40) : (!jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %42 = "jlir.ifelse"(%14, %39, %41) : (!jlir.Bool, !jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %43 = "jlir.or_int"(%37, %42) : (!jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %44 = "jlir.setfield!"(%0, %1, %31) : (!jlir.Task, !jlir.Symbol, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %45 = "jlir.setfield!"(%0, %3, %30) : (!jlir.Task, !jlir.Symbol, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %46 = "jlir.setfield!"(%0, %5, %32) : (!jlir.Task, !jlir.Symbol, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %47 = "jlir.setfield!"(%0, %7, %43) : (!jlir.Task, !jlir.Symbol, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %48 = "jlir.constant"() {value = #jlir<"0x0000000000000038">} : () -> !jlir.UInt64
# CHECK-NEXT:     %49 = "jlir.lshr_int"(%22, %48) : (!jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %50 = "jlir.constant"() {value = #jlir<"0xffffffffffffffc8">} : () -> !jlir.UInt64
# CHECK-NEXT:     %51 = "jlir.shl_int"(%22, %50) : (!jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %52 = "jlir.ifelse"(%14, %49, %51) : (!jlir.Bool, !jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %53 = "jlir.constant"() {value = #jlir<"0x0000000000000001">} : () -> !jlir.UInt64
# CHECK-NEXT:     %54 = "jlir.and_int"(%52, %53) : (!jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %55 = "jlir.constant"() {value = #jlir<"0x0000000000000000">} : () -> !jlir.UInt64
# CHECK-NEXT:     %56 = "jlir.==="(%54, %55) : (!jlir.UInt64, !jlir.UInt64) -> !jlir.Bool
# CHECK-NEXT:     %57 = "jlir.and_int"(%14, %56) : (!jlir.Bool, !jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     %58 = "jlir.not_int"(%57) : (!jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%58)[^bb3, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb2:  // pred: ^bb1
# CHECK-NEXT:     %59 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%14, %59)[^bb4] : (!jlir.Bool, !jlir.Bool) -> ()
# CHECK-NEXT:   ^bb3:  // pred: ^bb1
# CHECK-NEXT:     %60 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%60, %14)[^bb4] : (!jlir.Bool, !jlir.Bool) -> ()
# CHECK-NEXT:   ^bb4(%61: !jlir.Bool, %62: !jlir.Bool):  // 2 preds: ^bb2, ^bb3
# CHECK-NEXT:     "jlir.gotoifnot"(%61)[^bb6, ^bb5] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb5:  // pred: ^bb4
# CHECK-NEXT:     %63 = "jlir.constant"() {value = #jlir<"2">} : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%63)[^bb9] : (!jlir.Int64) -> ()
# CHECK-NEXT:   ^bb6:  // pred: ^bb4
# CHECK-NEXT:     "jlir.gotoifnot"(%62)[^bb8, ^bb7] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb7:  // pred: ^bb6
# CHECK-NEXT:     %64 = "jlir.constant"() {value = #jlir<"0">} : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%64)[^bb9] : (!jlir.Int64) -> ()
# CHECK-NEXT:   ^bb8:  // pred: ^bb6
# CHECK-NEXT:     %65 = "jlir.constant"() {value = #jlir<"ErrorException(\22fatal error in type inference (type bound)\22)">} : () -> !jlir.ErrorException
# CHECK-NEXT:     %66 = "jlir.throw"(%65) : (!jlir.ErrorException) -> !jlir<"Union{}">
# CHECK-NEXT:     %67 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.return"(%67) : (!jlir.Int64) -> ()
# CHECK-NEXT:   ^bb9(%68: !jlir.Int64):  // 2 preds: ^bb5, ^bb7
# CHECK-NEXT:     "jlir.return"(%68) : (!jlir.Int64) -> ()
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK: error: lowering to LLVM dialect failed
# CHECK-NEXT: error: module verification failed
