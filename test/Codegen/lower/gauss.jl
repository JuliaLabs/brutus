# RUN: julia -e "import Brutus; Brutus.lit(:emit_lowered)" --startup-file=no %s 2>&1 | FileCheck %s

function gauss(N)
    acc = 0
    for i in 1:N
        acc += i
    end
    return acc
end
emit(gauss, Int64)




# CHECK: Core.MethodMatch(Tuple{typeof(Main.Main.gauss), Int64}, svec(), gauss(N) in Main.Main at /home/mccoy/Dev/brutus/test/Codegen/lower/gauss.jl:3, true)after translating to MLIR in JLIR dialect:module  {
# CHECK-NEXT:   func nested @"Tuple{typeof(Main.gauss), Int64}"(%arg0: !jlir<"typeof(Main.gauss)">, %arg1: !jlir.Int64) -> !jlir.Int64 attributes {llvm.emit_c_interface} {
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

# CHECK: module  {
# CHECK-NEXT:   func nested @"Tuple{typeof(Main.gauss), Int64}"(%arg0: !jlir<"typeof(Main.gauss)">, %arg1: i64) -> i64 attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     %c1_i64 = constant 1 : i64
# CHECK-NEXT:     %c0_i64 = constant 0 : i64
# CHECK-NEXT:     %false = constant false
# CHECK-NEXT:     %true = constant true
# CHECK-NEXT:     %0 = cmpi sle, %c1_i64, %arg1 : i64
# CHECK-NEXT:     %1 = select %0, %arg1, %c0_i64 : i64
# CHECK-NEXT:     %2 = "jlir.convertstd"(%1) : (i64) -> !jlir.Int64
# CHECK-NEXT:     %3 = cmpi slt, %1, %c1_i64 : i64
# CHECK-NEXT:     cond_br %3, ^bb1, ^bb2(%false, %c1_i64, %c1_i64 : i1, i64, i64)
# CHECK-NEXT:   ^bb1:  // pred: ^bb0
# CHECK-NEXT:     %4 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %5 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %6 = "jlir.convertstd"(%4) : (!jlir.Int64) -> i64
# CHECK-NEXT:     %7 = "jlir.convertstd"(%5) : (!jlir.Int64) -> i64
# CHECK-NEXT:     br ^bb2(%true, %6, %7 : i1, i64, i64)
# CHECK-NEXT:   ^bb2(%8: i1, %9: i64, %10: i64):  // 2 preds: ^bb0, ^bb1
# CHECK-NEXT:     %11 = xor %8, %true : i1
# CHECK-NEXT:     cond_br %11, ^bb3(%9, %10, %c0_i64 : i64, i64, i64), ^bb7(%c0_i64 : i64)
# CHECK-NEXT:   ^bb3(%12: i64, %13: i64, %14: i64):  // 2 preds: ^bb2, ^bb6
# CHECK-NEXT:     %15 = "jlir.convertstd"(%13) : (i64) -> !jlir.Int64
# CHECK-NEXT:     %16 = addi %14, %12 : i64
# CHECK-NEXT:     %17 = "jlir.==="(%15, %2) : (!jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     %18 = "jlir.convertstd"(%17) : (!jlir.Bool) -> i1
# CHECK-NEXT:     cond_br %18, ^bb4, ^bb5
# CHECK-NEXT:   ^bb4:  // pred: ^bb3
# CHECK-NEXT:     %19 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %20 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %21 = "jlir.convertstd"(%19) : (!jlir.Int64) -> i64
# CHECK-NEXT:     %22 = "jlir.convertstd"(%20) : (!jlir.Int64) -> i64
# CHECK-NEXT:     br ^bb6(%21, %22, %true : i64, i64, i1)
# CHECK-NEXT:   ^bb5:  // pred: ^bb3
# CHECK-NEXT:     %23 = addi %13, %c1_i64 : i64
# CHECK-NEXT:     br ^bb6(%23, %23, %false : i64, i64, i1)
# CHECK-NEXT:   ^bb6(%24: i64, %25: i64, %26: i1):  // 2 preds: ^bb4, ^bb5
# CHECK-NEXT:     %27 = xor %26, %true : i1
# CHECK-NEXT:     cond_br %27, ^bb3(%24, %25, %16 : i64, i64, i64), ^bb7(%16 : i64)
# CHECK-NEXT:   ^bb7(%28: i64):  // 2 preds: ^bb2, ^bb6
# CHECK-NEXT:     return %28 : i64
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK:   llvm.func @"Tuple{typeof(Main.gauss), Int64}"(%arg0: !llvm.ptr<struct<"struct_jl_value_type", opaque>>, %arg1: i64) -> i64 attributes {llvm.emit_c_interface, sym_visibility = "nested"} {
# CHECK-NEXT:     %0 = llvm.mlir.constant({{[0-9]+}} : i64) : i64
# CHECK-NEXT:     %1 = llvm.mlir.constant({{[0-9]+}} : i64) : i64
# CHECK-NEXT:     %2 = llvm.mlir.constant(false) : i1
# CHECK-NEXT:     %3 = llvm.mlir.constant(true) : i1
# CHECK-NEXT:     %4 = llvm.icmp "sle" %0, %arg1 : i64
# CHECK-NEXT:     %5 = llvm.select %4, %arg1, %1 : i1, i64
# CHECK-NEXT:     %6 = llvm.icmp "slt" %5, %0 : i64
# CHECK-NEXT:     llvm.cond_br %6, ^bb1, ^bb2(%2, %0, %0 : i1, i64, i64)
# CHECK-NEXT:   ^bb1:  // pred: ^bb0
# CHECK-NEXT:     %7 = llvm.mlir.undef : i64
# CHECK-NEXT:     llvm.br ^bb2(%3, %7, %7 : i1, i64, i64)
# CHECK-NEXT:   ^bb2(%8: i1, %9: i64, %10: i64):  // 2 preds: ^bb0, ^bb1
# CHECK-NEXT:     %11 = llvm.xor %8, %3  : i1
# CHECK-NEXT:     llvm.cond_br %11, ^bb3(%9, %10, %1 : i64, i64, i64), ^bb7(%1 : i64)
# CHECK-NEXT:   ^bb3(%12: i64, %13: i64, %14: i64):  // 2 preds: ^bb2, ^bb6
# CHECK-NEXT:     %15 = llvm.add %14, %12  : i64
# CHECK-NEXT:     %16 = llvm.icmp "eq" %13, %5 : i64
# CHECK-NEXT:     llvm.cond_br %16, ^bb4, ^bb5
# CHECK-NEXT:   ^bb4:  // pred: ^bb3
# CHECK-NEXT:     %17 = llvm.mlir.undef : i64
# CHECK-NEXT:     llvm.br ^bb6(%17, %17, %3 : i64, i64, i1)
# CHECK-NEXT:   ^bb5:  // pred: ^bb3
# CHECK-NEXT:     %18 = llvm.add %13, %0  : i64
# CHECK-NEXT:     llvm.br ^bb6(%18, %18, %2 : i64, i64, i1)
# CHECK-NEXT:   ^bb6(%19: i64, %20: i64, %21: i1):  // 2 preds: ^bb4, ^bb5
# CHECK-NEXT:     %22 = llvm.xor %21, %3  : i1
# CHECK-NEXT:     llvm.cond_br %22, ^bb3(%19, %20, %15 : i64, i64, i64), ^bb7(%15 : i64)
# CHECK-NEXT:   ^bb7(%23: i64):  // 2 preds: ^bb2, ^bb6
# CHECK-NEXT:     llvm.return %23 : i64
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @"_mlir_ciface_Tuple{typeof(Main.gauss), Int64}"(%arg0: !llvm.ptr<struct<"struct_jl_value_type", opaque>>, %arg1: i64) -> i64 attributes {llvm.emit_c_interface, sym_visibility = "nested"} {
# CHECK-NEXT:     %0 = llvm.call @"Tuple{typeof(Main.gauss), Int64}"(%arg0, %arg1) : (!llvm.ptr<struct<"struct_jl_value_type", opaque>>, i64) -> i64
# CHECK-NEXT:     llvm.return %0 : i64
# CHECK-NEXT:   }
# CHECK-NEXT: }
