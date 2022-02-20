# RUN: julia -e "import Brutus; Brutus.lit(:emit_lowered)" --startup-file=no %s 2>&1 | FileCheck %s

function gauss(N)
    acc = 0
    for i in 1:N
        acc += i
    end
    return acc
end
emit(gauss, Int64)




# CHECK: module  {
# CHECK-NEXT:   func nested @"Tuple{typeof(Main.gauss), Int64}"(%arg0: !jlir<"typeof(Main.gauss)">, %arg1: !jlir.Int64, %arg2: !jlir<"Union{Nothing, Tuple{Int64, Int64}}">, %arg3: !jlir.Int64, %arg4: !jlir.Int64) -> !jlir.Int64 attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     "jlir.goto"()[^bb1] : () -> ()
# CHECK-NEXT:   ^bb1:  // pred: ^bb0
# CHECK-NEXT:     %0 = "jlir.constant"() {value = #jlir<"#<intrinsic #29 sle_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %1 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %2 = "jlir.call"(%0, %1, %arg1) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%2)[^bb3, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb2:  // pred: ^bb1
# CHECK-NEXT:     "jlir.goto"(%arg1)[^bb4] : (!jlir.Int64) -> ()
# CHECK-NEXT:   ^bb3:  // pred: ^bb1
# CHECK-NEXT:     %3 = "jlir.constant"() {value = #jlir<"0">} : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%3)[^bb4] : (!jlir.Int64) -> ()
# CHECK-NEXT:   ^bb4(%4: !jlir.Int64):  // 2 preds: ^bb2, ^bb3
# CHECK-NEXT:     "jlir.goto"()[^bb5] : () -> ()
# CHECK-NEXT:   ^bb5:  // pred: ^bb4
# CHECK-NEXT:     "jlir.goto"()[^bb6] : () -> ()
# CHECK-NEXT:   ^bb6:  // pred: ^bb5
# CHECK-NEXT:     %5 = "jlir.constant"() {value = #jlir<"#<intrinsic #27 slt_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %6 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %7 = "jlir.call"(%5, %4, %6) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%7)[^bb8, ^bb7] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb7:  // pred: ^bb6
# CHECK-NEXT:     %8 = "jlir.constant"() {value = #jlir.nothing} : () -> !jlir.Nothing
# CHECK-NEXT:     %9 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     %10 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %11 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%9, %10, %11)[^bb9] : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb8:  // pred: ^bb6
# CHECK-NEXT:     %12 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK-NEXT:     %13 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %14 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%12, %13, %14)[^bb9] : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb9(%15: !jlir.Bool, %16: !jlir.Int64, %17: !jlir.Int64):  // 2 preds: ^bb7, ^bb8
# CHECK-NEXT:     %18 = "jlir.constant"() {value = #jlir<"#<intrinsic #43 not_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %19 = "jlir.call"(%18, %15) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     %20 = "jlir.constant"() {value = #jlir<"0">} : () -> !jlir.Int64
# CHECK-NEXT:     %21 = "jlir.constant"() {value = #jlir<"0">} : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.gotoifnot"(%19, %21, %16, %17, %20)[^bb15, ^bb10] {operand_segment_sizes = dense<[1, 1, 3]> : vector<3xi32>} : (!jlir.Bool, !jlir.Int64, !jlir.Int64, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb10(%22: !jlir.Int64, %23: !jlir.Int64, %24: !jlir.Int64):  // 2 preds: ^bb9, ^bb14
# CHECK-NEXT:     %25 = "jlir.constant"() {value = #jlir<"#<intrinsic #2 add_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %26 = "jlir.call"(%25, %24, %22) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %27 = "jlir.constant"() {value = #jlir<"===">} : () -> !jlir<"typeof(===)">
# CHECK-NEXT:     %28 = "jlir.call"(%27, %23, %4) : (!jlir<"typeof(===)">, !jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%28)[^bb12, ^bb11] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb11:  // pred: ^bb10
# CHECK-NEXT:     %29 = "jlir.constant"() {value = #jlir.nothing} : () -> !jlir.Nothing
# CHECK-NEXT:     %30 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %31 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %32 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%30, %31, %32)[^bb13] : (!jlir.Int64, !jlir.Int64, !jlir.Bool) -> ()
# CHECK-NEXT:   ^bb12:  // pred: ^bb10
# CHECK-NEXT:     %33 = "jlir.constant"() {value = #jlir<"#<intrinsic #2 add_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %34 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %35 = "jlir.call"(%33, %23, %34) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     %36 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK-NEXT:     "jlir.goto"(%35, %35, %36)[^bb13] : (!jlir.Int64, !jlir.Int64, !jlir.Bool) -> ()
# CHECK-NEXT:   ^bb13(%37: !jlir.Int64, %38: !jlir.Int64, %39: !jlir.Bool):  // 2 preds: ^bb11, ^bb12
# CHECK-NEXT:     %40 = "jlir.constant"() {value = #jlir<"#<intrinsic #43 not_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %41 = "jlir.call"(%40, %39) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     "jlir.gotoifnot"(%41, %26)[^bb15, ^bb14] {operand_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>} : (!jlir.Bool, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb14:  // pred: ^bb13
# CHECK-NEXT:     "jlir.goto"(%37, %38, %26)[^bb10] : (!jlir.Int64, !jlir.Int64, !jlir.Int64) -> ()
# CHECK-NEXT:   ^bb15(%42: !jlir.Int64):  // 2 preds: ^bb9, ^bb13
# CHECK-NEXT:     "jlir.return"(%42) : (!jlir.Int64) -> ()
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK: module  {
# CHECK-NEXT:   func nested @"Tuple{typeof(Main.gauss), Int64}"(%arg0: !jlir<"typeof(Main.gauss)">, %arg1: i64, %arg2: !jlir<"Union{Nothing, Tuple{Int64, Int64}}">, %arg3: i64, %arg4: i64) -> i64 attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     %false = constant false
# CHECK-NEXT:     %true = constant true
# CHECK-NEXT:     %c0_i64 = constant 0 : i64
# CHECK-NEXT:     %c1_i64 = constant 1 : i64
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

# CHECK: module  {
# CHECK-NEXT:   llvm.func @"Tuple{typeof(Main.gauss), Int64}"(%arg0: !llvm.ptr<struct<"struct_jl_value_type", opaque>>, %arg1: i64, %arg2: !llvm.ptr<struct<"struct_jl_value_type", opaque>>, %arg3: i64, %arg4: i64) -> i64 attributes {llvm.emit_c_interface, sym_visibility = "nested"} {
# CHECK-NEXT:     %0 = llvm.mlir.constant(false) : i1
# CHECK-NEXT:     %1 = llvm.mlir.constant(true) : i1
# CHECK-NEXT:     %2 = llvm.mlir.constant({{[0-9]+}} : i64) : i64
# CHECK-NEXT:     %3 = llvm.mlir.constant({{[0-9]+}} : i64) : i64
# CHECK-NEXT:     %4 = llvm.icmp "sle" %3, %arg1 : i64
# CHECK-NEXT:     %5 = llvm.select %4, %arg1, %2 : i1, i64
# CHECK-NEXT:     %6 = llvm.icmp "slt" %5, %3 : i64
# CHECK-NEXT:     llvm.cond_br %6, ^bb1, ^bb2(%0, %3, %3 : i1, i64, i64)
# CHECK-NEXT:   ^bb1:  // pred: ^bb0
# CHECK-NEXT:     %7 = llvm.mlir.undef : i64
# CHECK-NEXT:     llvm.br ^bb2(%1, %7, %7 : i1, i64, i64)
# CHECK-NEXT:   ^bb2(%8: i1, %9: i64, %10: i64):  // 2 preds: ^bb0, ^bb1
# CHECK-NEXT:     %11 = llvm.xor %8, %1  : i1
# CHECK-NEXT:     llvm.cond_br %11, ^bb3(%9, %10, %2 : i64, i64, i64), ^bb7(%2 : i64)
# CHECK-NEXT:   ^bb3(%12: i64, %13: i64, %14: i64):  // 2 preds: ^bb2, ^bb6
# CHECK-NEXT:     %15 = llvm.add %14, %12  : i64
# CHECK-NEXT:     %16 = llvm.icmp "eq" %13, %5 : i64
# CHECK-NEXT:     llvm.cond_br %16, ^bb4, ^bb5
# CHECK-NEXT:   ^bb4:  // pred: ^bb3
# CHECK-NEXT:     %17 = llvm.mlir.undef : i64
# CHECK-NEXT:     llvm.br ^bb6(%17, %17, %1 : i64, i64, i1)
# CHECK-NEXT:   ^bb5:  // pred: ^bb3
# CHECK-NEXT:     %18 = llvm.add %13, %3  : i64
# CHECK-NEXT:     llvm.br ^bb6(%18, %18, %0 : i64, i64, i1)
# CHECK-NEXT:   ^bb6(%19: i64, %20: i64, %21: i1):  // 2 preds: ^bb4, ^bb5
# CHECK-NEXT:     %22 = llvm.xor %21, %1  : i1
# CHECK-NEXT:     llvm.cond_br %22, ^bb3(%19, %20, %15 : i64, i64, i64), ^bb7(%15 : i64)
# CHECK-NEXT:   ^bb7(%23: i64):  // 2 preds: ^bb2, ^bb6
# CHECK-NEXT:     llvm.return %23 : i64
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @"_mlir_ciface_Tuple{typeof(Main.gauss), Int64}"(%arg0: !llvm.ptr<struct<"struct_jl_value_type", opaque>>, %arg1: i64, %arg2: !llvm.ptr<struct<"struct_jl_value_type", opaque>>, %arg3: i64, %arg4: i64) -> i64 attributes {llvm.emit_c_interface, sym_visibility = "nested"} {
# CHECK-NEXT:     %0 = llvm.call @"Tuple{typeof(Main.gauss), Int64}"(%arg0, %arg1, %arg2, %arg3, %arg4) : (!llvm.ptr<struct<"struct_jl_value_type", opaque>>, i64, !llvm.ptr<struct<"struct_jl_value_type", opaque>>, i64, i64) -> i64
# CHECK-NEXT:     llvm.return %0 : i64
# CHECK-NEXT:   }
# CHECK-NEXT: }
