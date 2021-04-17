# RUN: julia -e "import Brutus; Brutus.lit(:emit_optimized)" --startup-file=no %s 2>&1 | FileCheck %s

function calls()
    f = rand(Bool) ? (+) : (-)
    return f(1, 1)
end
emit(calls)



# CHECK:   func nested @"Tuple{typeof(Main.calls)}"(%arg0: !jlir<"typeof(Main.calls)">) -> !jlir.Int64 attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     "jlir.goto"()[^bb1] : () -> ()
# CHECK-NEXT:   ^bb1:  // pred: ^bb0
# CHECK-NEXT:     %0 = "jlir.unimplemented"() : () -> !jlir.Int16
# CHECK-NEXT:     %1 = "jlir.constant"() {value = #jlir<"#<intrinsic #52 sext_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %2 = "jlir.constant"() {value = #jlir.Int64} : () -> !jlir.DataType
# CHECK-NEXT:     %3 = "jlir.call"(%1, %2, %0) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.DataType, !jlir.Int16) -> !jlir.Int64
# CHECK-NEXT:     %4 = "jlir.constant"() {value = #jlir<"#<intrinsic #2 add_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %5 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %6 = "jlir.call"(%4, %3, %5) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %7 = "jlir.constant"() {value = #jlir.Random.default_rng} : () -> !jlir<"typeof(Random.default_rng)">
# CHECK-NEXT:     %8 = "jlir.invoke"(%7, %6) {methodInstance = #jlir<"default_rng(Int64) from default_rng(Int64)">} : (!jlir<"typeof(Random.default_rng)">, !jlir.Int64) -> !jlir.Random.MersenneTwister
# CHECK-NEXT:     %9 = "jlir.constant"() {value = #jlir.getfield} : () -> !jlir<"typeof(getfield)">
# CHECK-NEXT:     %10 = "jlir.constant"() {value = #jlir<":idxF">} : () -> !jlir.Symbol
# CHECK-NEXT:     %11 = "jlir.call"(%9, %8, %10) : (!jlir<"typeof(getfield)">, !jlir.Random.MersenneTwister, !jlir.Symbol) -> !jlir.Int64
# CHECK-NEXT:     %12 = "jlir.constant"() {value = #jlir<"1002">} : () -> !jlir.Int64
# CHECK-NEXT:     %13 = "jlir.constant"() {value = #jlir<"===">} : () -> !jlir<"typeof(===)">
# CHECK-NEXT:     %14 = "jlir.call"(%13, %11, %12) : (!jlir<"typeof(===)">, !jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%14)[^bb3, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb2:  // pred: ^bb1
# CHECK-NEXT:     %15 = "jlir.constant"() {value = #jlir.Random.gen_rand} : () -> !jlir<"typeof(Random.gen_rand)">
# CHECK-NEXT:     %16 = "jlir.invoke"(%15, %8) {methodInstance = #jlir<"gen_rand(Random.MersenneTwister) from gen_rand(Random.MersenneTwister)">} : (!jlir<"typeof(Random.gen_rand)">, !jlir.Random.MersenneTwister) -> !jlir.Any
# CHECK-NEXT:     "jlir.goto"()[^bb3] : () -> ()
# CHECK-NEXT:   ^bb3:  // 2 preds: ^bb1, ^bb2
# CHECK-NEXT:     "jlir.goto"()[^bb4] : () -> ()
# CHECK-NEXT:   ^bb4:  // pred: ^bb3
# CHECK-NEXT:     %17 = "jlir.constant"() {value = #jlir.getfield} : () -> !jlir<"typeof(getfield)">
# CHECK-NEXT:     %18 = "jlir.constant"() {value = #jlir<":vals">} : () -> !jlir.Symbol
# CHECK-NEXT:     %19 = "jlir.call"(%17, %8, %18) : (!jlir<"typeof(getfield)">, !jlir.Random.MersenneTwister, !jlir.Symbol) -> !jlir<"Array{Float64, 1}">
# CHECK-NEXT:     %20 = "jlir.constant"() {value = #jlir.getfield} : () -> !jlir<"typeof(getfield)">
# CHECK-NEXT:     %21 = "jlir.constant"() {value = #jlir<":idxF">} : () -> !jlir.Symbol
# CHECK-NEXT:     %22 = "jlir.call"(%20, %8, %21) : (!jlir<"typeof(getfield)">, !jlir.Random.MersenneTwister, !jlir.Symbol) -> !jlir.Int64
# CHECK-NEXT:     %23 = "jlir.constant"() {value = #jlir<"#<intrinsic #2 add_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %24 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %25 = "jlir.call"(%23, %22, %24) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %26 = "jlir.constant"() {value = #jlir<"setfield!">} : () -> !jlir<"typeof(setfield!)">
# CHECK-NEXT:     %27 = "jlir.constant"() {value = #jlir<":idxF">} : () -> !jlir.Symbol
# CHECK-NEXT:     %28 = "jlir.call"(%26, %8, %27, %25) : (!jlir<"typeof(setfield!)">, !jlir.Random.MersenneTwister, !jlir.Symbol, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %29 = "jlir.constant"() {value = #jlir.Core.arrayref} : () -> !jlir<"typeof(Core.arrayref)">
# CHECK-NEXT:     %30 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK-NEXT:     %31 = "jlir.call"(%29, %30, %19, %25) : (!jlir<"typeof(Core.arrayref)">, !jlir.Bool, !jlir<"Array{Float64, 1}">, !jlir.Int64) -> !jlir.Float64
# CHECK-NEXT:     %32 = "jlir.constant"() {value = #jlir<"#<intrinsic #0 bitcast>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %33 = "jlir.constant"() {value = #jlir.UInt64} : () -> !jlir.DataType
# CHECK-NEXT:     %34 = "jlir.call"(%32, %33, %31) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.DataType, !jlir.Float64) -> !jlir.UInt64
# CHECK-NEXT:     "jlir.goto"()[^bb5] : () -> ()
# CHECK-NEXT:   ^bb5:  // pred: ^bb4
# CHECK-NEXT:     "jlir.goto"()[^bb6] : () -> ()
# CHECK-NEXT:   ^bb6:  // pred: ^bb5
# CHECK-NEXT:     %35 = "jlir.constant"() {value = #jlir<"#<intrinsic #41 and_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %36 = "jlir.constant"() {value = #jlir<"0x0000000000000001">} : () -> !jlir.UInt64
# CHECK-NEXT:     %37 = "jlir.call"(%35, %34, %36) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %38 = "jlir.constant"() {value = #jlir<"#<intrinsic #29 sle_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %39 = "jlir.constant"() {value = #jlir<"0">} : () -> !jlir.Int64
# CHECK-NEXT:     %40 = "jlir.constant"() {value = #jlir<"0">} : () -> !jlir.Int64
# CHECK-NEXT:     %41 = "jlir.call"(%38, %39, %40) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     %42 = "jlir.constant"() {value = #jlir<"#<intrinsic #0 bitcast>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %43 = "jlir.constant"() {value = #jlir.UInt64} : () -> !jlir.DataType
# CHECK-NEXT:     %44 = "jlir.constant"() {value = #jlir<"0">} : () -> !jlir.Int64
# CHECK-NEXT:     %45 = "jlir.call"(%42, %43, %44) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.DataType, !jlir.Int64) -> !jlir.UInt64
# CHECK-NEXT:     %46 = "jlir.constant"() {value = #jlir<"===">} : () -> !jlir<"typeof(===)">
# CHECK-NEXT:     %47 = "jlir.call"(%46, %37, %45) : (!jlir<"typeof(===)">, !jlir.UInt64, !jlir.UInt64) -> !jlir.Bool
# CHECK-NEXT:     %48 = "jlir.constant"() {value = #jlir<"#<intrinsic #41 and_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %49 = "jlir.call"(%48, %41, %47) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Bool, !jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     %50 = "jlir.constant"() {value = #jlir<"#<intrinsic #44 not_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %51 = "jlir.call"(%50, %49) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"()[^bb7] : () -> ()
# CHECK-NEXT:   ^bb7:  // pred: ^bb6
# CHECK-NEXT:     "jlir.goto"()[^bb8] : () -> ()
# CHECK-NEXT:   ^bb8:  // pred: ^bb7
# CHECK-NEXT:     "jlir.goto"()[^bb9] : () -> ()
# CHECK-NEXT:   ^bb9:  // pred: ^bb8
# CHECK-NEXT:     "jlir.gotoifnot"(%51)[^bb11, ^bb10] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb10:  // pred: ^bb9
# CHECK-NEXT:     %52 = "jlir.constant"() {value = #jlir<"Base.:(+)">} : () -> !jlir<"typeof(Base.:(+))">
# CHECK-NEXT:     %53 = "jlir.pi"(%52) : (!jlir<"typeof(Base.:(+))">) -> !jlir<"Union{typeof(Base.:(+)), typeof(Base.:(-))}">
# CHECK-NEXT:     "jlir.goto"(%53)[^bb12] : (!jlir<"Union{typeof(Base.:(+)), typeof(Base.:(-))}">) -> ()
# CHECK-NEXT:   ^bb11:  // pred: ^bb9
# CHECK-NEXT:     %54 = "jlir.constant"() {value = #jlir<"Base.:(-)">} : () -> !jlir<"typeof(Base.:(-))">
# CHECK-NEXT:     %55 = "jlir.pi"(%54) : (!jlir<"typeof(Base.:(-))">) -> !jlir<"Union{typeof(Base.:(+)), typeof(Base.:(-))}">
# CHECK-NEXT:     "jlir.goto"(%55)[^bb12] : (!jlir<"Union{typeof(Base.:(+)), typeof(Base.:(-))}">) -> ()
# CHECK-NEXT:   ^bb12(%56: !jlir<"Union{typeof(Base.:(+)), typeof(Base.:(-))}">):  // 2 preds: ^bb10, ^bb11
# CHECK-NEXT:     %57 = "jlir.constant"() {value = #jlir.isa} : () -> !jlir<"typeof(isa)">
# CHECK-NEXT:     %58 = "jlir.constant"() {value = #jlir<"typeof(Base.:(+))">} : () -> !jlir.DataType
# CHECK-NEXT:     %59 = "jlir.call"(%57, %56, %58) : (!jlir<"typeof(isa)">, !jlir<"Union{typeof(Base.:(+)), typeof(Base.:(-))}">, !jlir.DataType) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%59)[^bb14, ^bb13] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb13:  // pred: ^bb12
# CHECK-NEXT:     %60 = "jlir.constant"() {value = #jlir<"#<intrinsic #2 add_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %61 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %62 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %63 = "jlir.call"(%60, %61, %62) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%63)[^bb17] : (!jlir.Int64) -> ()
# CHECK-NEXT:   ^bb14:  // pred: ^bb12
# CHECK-NEXT:     %64 = "jlir.constant"() {value = #jlir.isa} : () -> !jlir<"typeof(isa)">
# CHECK-NEXT:     %65 = "jlir.constant"() {value = #jlir<"typeof(Base.:(-))">} : () -> !jlir.DataType
# CHECK-NEXT:     %66 = "jlir.call"(%64, %56, %65) : (!jlir<"typeof(isa)">, !jlir<"Union{typeof(Base.:(+)), typeof(Base.:(-))}">, !jlir.DataType) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%66)[^bb16, ^bb15] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb15:  // pred: ^bb14
# CHECK-NEXT:     %67 = "jlir.constant"() {value = #jlir<"#<intrinsic #3 sub_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %68 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %69 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %70 = "jlir.call"(%67, %68, %69) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%70)[^bb17] : (!jlir.Int64) -> ()
# CHECK-NEXT:   ^bb16:  // pred: ^bb14
# CHECK-NEXT:     %71 = "jlir.constant"() {value = #jlir.throw} : () -> !jlir<"typeof(throw)">
# CHECK-NEXT:     %72 = "jlir.constant"() {value = #jlir<"ErrorException("fatal error in type inference (type bound)")">} : () -> !jlir.ErrorException
# CHECK-NEXT:     %73 = "jlir.call"(%71, %72) : (!jlir<"typeof(throw)">, !jlir.ErrorException) -> !jlir<"Union{}">
# CHECK-NEXT:     %74 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.return"(%74) : (!jlir.Int64) -> ()
# CHECK-NEXT:   ^bb17(%75: !jlir.Int64):  // 2 preds: ^bb13, ^bb15
# CHECK-NEXT:     "jlir.return"(%75) : (!jlir.Int64) -> ()
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK:   func nested @"Tuple{typeof(Main.calls)}"(%arg0: !jlir<"typeof(Main.calls)">) -> !jlir.Int64 attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     "jlir.goto"()[^bb1] : () -> ()
# CHECK-NEXT:   ^bb1:  // pred: ^bb0
# CHECK-NEXT:     %0 = "jlir.unimplemented"() : () -> !jlir.Int16
# CHECK-NEXT:     %1 = "jlir.constant"() {value = #jlir.Int64} : () -> !jlir.DataType
# CHECK-NEXT:     %2 = "jlir.sext_int"(%1, %0) : (!jlir.DataType, !jlir.Int16) -> !jlir.Int64
# CHECK-NEXT:     %3 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %4 = "jlir.add_int"(%2, %3) : (!jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %5 = "jlir.constant"() {value = #jlir.Random.default_rng} : () -> !jlir<"typeof(Random.default_rng)">
# CHECK-NEXT:     %6 = "jlir.invoke"(%5, %4) {methodInstance = #jlir<"default_rng(Int64) from default_rng(Int64)">} : (!jlir<"typeof(Random.default_rng)">, !jlir.Int64) -> !jlir.Random.MersenneTwister
# CHECK-NEXT:     %7 = "jlir.constant"() {value = #jlir<":idxF">} : () -> !jlir.Symbol
# CHECK-NEXT:     %8 = "jlir.getfield"(%6, %7) : (!jlir.Random.MersenneTwister, !jlir.Symbol) -> !jlir.Int64
# CHECK-NEXT:     %9 = "jlir.constant"() {value = #jlir<"1002">} : () -> !jlir.Int64
# CHECK-NEXT:     %10 = "jlir.==="(%8, %9) : (!jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%10)[^bb3, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb2:  // pred: ^bb1
# CHECK-NEXT:     %11 = "jlir.constant"() {value = #jlir.Random.gen_rand} : () -> !jlir<"typeof(Random.gen_rand)">
# CHECK-NEXT:     %12 = "jlir.invoke"(%11, %6) {methodInstance = #jlir<"gen_rand(Random.MersenneTwister) from gen_rand(Random.MersenneTwister)">} : (!jlir<"typeof(Random.gen_rand)">, !jlir.Random.MersenneTwister) -> !jlir.Any
# CHECK-NEXT:     "jlir.goto"()[^bb3] : () -> ()
# CHECK-NEXT:   ^bb3:  // 2 preds: ^bb1, ^bb2
# CHECK-NEXT:     "jlir.goto"()[^bb4] : () -> ()
# CHECK-NEXT:   ^bb4:  // pred: ^bb3
# CHECK-NEXT:     %13 = "jlir.constant"() {value = #jlir<":vals">} : () -> !jlir.Symbol
# CHECK-NEXT:     %14 = "jlir.getfield"(%6, %13) : (!jlir.Random.MersenneTwister, !jlir.Symbol) -> !jlir<"Array{Float64, 1}">
# CHECK-NEXT:     %15 = "jlir.getfield"(%6, %7) : (!jlir.Random.MersenneTwister, !jlir.Symbol) -> !jlir.Int64
# CHECK-NEXT:     %16 = "jlir.add_int"(%15, %3) : (!jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %17 = "jlir.setfield!"(%6, %7, %16) : (!jlir.Random.MersenneTwister, !jlir.Symbol, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %18 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK-NEXT:     %19 = "jlir.arrayref"(%18, %14, %16) : (!jlir.Bool, !jlir<"Array{Float64, 1}">, !jlir.Int64) -> !jlir.Float64
# CHECK-NEXT:     %20 = "jlir.constant"() {value = #jlir.UInt64} : () -> !jlir.DataType
# CHECK-NEXT:     %21 = "jlir.bitcast"(%20, %19) : (!jlir.DataType, !jlir.Float64) -> !jlir.UInt64
# CHECK-NEXT:     "jlir.goto"()[^bb5] : () -> ()
# CHECK-NEXT:   ^bb5:  // pred: ^bb4
# CHECK-NEXT:     "jlir.goto"()[^bb6] : () -> ()
# CHECK-NEXT:   ^bb6:  // pred: ^bb5
# CHECK-NEXT:     %22 = "jlir.constant"() {value = #jlir<"0x0000000000000001">} : () -> !jlir.UInt64
# CHECK-NEXT:     %23 = "jlir.and_int"(%21, %22) : (!jlir.UInt64, !jlir.UInt64) -> !jlir.UInt64
# CHECK-NEXT:     %24 = "jlir.constant"() {value = #jlir<"0">} : () -> !jlir.Int64
# CHECK-NEXT:     %25 = "jlir.sle_int"(%24, %24) : (!jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     %26 = "jlir.bitcast"(%20, %24) : (!jlir.DataType, !jlir.Int64) -> !jlir.UInt64
# CHECK-NEXT:     %27 = "jlir.==="(%23, %26) : (!jlir.UInt64, !jlir.UInt64) -> !jlir.Bool
# CHECK-NEXT:     %28 = "jlir.and_int"(%25, %27) : (!jlir.Bool, !jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     %29 = "jlir.not_int"(%28) : (!jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"()[^bb7] : () -> ()
# CHECK-NEXT:   ^bb7:  // pred: ^bb6
# CHECK-NEXT:     "jlir.goto"()[^bb8] : () -> ()
# CHECK-NEXT:   ^bb8:  // pred: ^bb7
# CHECK-NEXT:     "jlir.goto"()[^bb9] : () -> ()
# CHECK-NEXT:   ^bb9:  // pred: ^bb8
# CHECK-NEXT:     "jlir.gotoifnot"(%29)[^bb11, ^bb10] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb10:  // pred: ^bb9
# CHECK-NEXT:     %30 = "jlir.constant"() {value = #jlir<"Base.:(+)">} : () -> !jlir<"typeof(Base.:(+))">
# CHECK-NEXT:     %31 = "jlir.pi"(%30) : (!jlir<"typeof(Base.:(+))">) -> !jlir<"Union{typeof(Base.:(+)), typeof(Base.:(-))}">
# CHECK-NEXT:     "jlir.goto"(%31)[^bb12] : (!jlir<"Union{typeof(Base.:(+)), typeof(Base.:(-))}">) -> ()
# CHECK-NEXT:   ^bb11:  // pred: ^bb9
# CHECK-NEXT:     %32 = "jlir.constant"() {value = #jlir<"Base.:(-)">} : () -> !jlir<"typeof(Base.:(-))">
# CHECK-NEXT:     %33 = "jlir.pi"(%32) : (!jlir<"typeof(Base.:(-))">) -> !jlir<"Union{typeof(Base.:(+)), typeof(Base.:(-))}">
# CHECK-NEXT:     "jlir.goto"(%33)[^bb12] : (!jlir<"Union{typeof(Base.:(+)), typeof(Base.:(-))}">) -> ()
# CHECK-NEXT:   ^bb12(%34: !jlir<"Union{typeof(Base.:(+)), typeof(Base.:(-))}">):  // 2 preds: ^bb10, ^bb11
# CHECK-NEXT:     %35 = "jlir.constant"() {value = #jlir<"typeof(Base.:(+))">} : () -> !jlir.DataType
# CHECK-NEXT:     %36 = "jlir.isa"(%34, %35) : (!jlir<"Union{typeof(Base.:(+)), typeof(Base.:(-))}">, !jlir.DataType) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%36)[^bb14, ^bb13] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb13:  // pred: ^bb12
# CHECK-NEXT:     %37 = "jlir.add_int"(%3, %3) : (!jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%37)[^bb17] : (!jlir.Int64) -> ()
# CHECK-NEXT:   ^bb14:  // pred: ^bb12
# CHECK-NEXT:     %38 = "jlir.constant"() {value = #jlir<"typeof(Base.:(-))">} : () -> !jlir.DataType
# CHECK-NEXT:     %39 = "jlir.isa"(%34, %38) : (!jlir<"Union{typeof(Base.:(+)), typeof(Base.:(-))}">, !jlir.DataType) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%39)[^bb16, ^bb15] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb15:  // pred: ^bb14
# CHECK-NEXT:     %40 = "jlir.sub_int"(%3, %3) : (!jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%40)[^bb17] : (!jlir.Int64) -> ()
# CHECK-NEXT:   ^bb16:  // pred: ^bb14
# CHECK-NEXT:     %41 = "jlir.constant"() {value = #jlir<"ErrorException("fatal error in type inference (type bound)")">} : () -> !jlir.ErrorException
# CHECK-NEXT:     %42 = "jlir.throw"(%41) : (!jlir.ErrorException) -> !jlir<"Union{}">
# CHECK-NEXT:     %43 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.return"(%43) : (!jlir.Int64) -> ()
# CHECK-NEXT:   ^bb17(%44: !jlir.Int64):  // 2 preds: ^bb13, ^bb15
# CHECK-NEXT:     "jlir.return"(%44) : (!jlir.Int64) -> ()
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK: error: lowering to Standard dialect failed

# CHECK: error: lowering to LLVM dialect failed
# CHECK-NEXT: error: module verification failed
