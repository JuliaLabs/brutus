# RUN: julia -e "import Brutus; Brutus.lit(:emit_lowered)" --startup-file=no %s 2>&1 | FileCheck %s

function gauss(N)
    acc = 0
    for i in 1:N
        acc += i
    end
    return acc
end
emit(gauss, Int64)



# CHECK: module {
# CHECK-NEXT:   func @"Tuple{typeof(Main.gauss), Int64}"(%arg0: !jlir<"typeof(Main.gauss)">, %arg1: i64) -> i64 {
# CHECK-NEXT:     %c1_i64 = constant 1 : i64
# CHECK-NEXT:     %c0_i64 = constant 0 : i64
# CHECK-NEXT:     %false = constant false
# CHECK-NEXT:     %true = constant true
# CHECK-NEXT:     %0 = cmpi "sle", %c1_i64, %arg1 : i64
# CHECK-NEXT:     %1 = select %0, %arg1, %c0_i64 : i64
# CHECK-NEXT:     %2 = "jlir.convertstd"(%1) : (i64) -> !jlir.Int64
# CHECK-NEXT:     %3 = cmpi "slt", %1, %c1_i64 : i64
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

# CHECK: module {
# CHECK-NEXT:   llvm.func @"Tuple{typeof(Main.gauss), Int64}"(%arg0: !llvm.ptr<struct<"jl_value_t", ()>>, %arg1: !llvm.i64) -> !llvm.i64 {
# CHECK-NEXT:     %0 = llvm.mlir.constant({{[0-9]+}} : i64) : !llvm.i64
# CHECK-NEXT:     %1 = llvm.mlir.constant({{[0-9]+}} : i64) : !llvm.i64
# CHECK-NEXT:     %2 = llvm.mlir.constant(false) : !llvm.i1
# CHECK-NEXT:     %3 = llvm.mlir.constant(true) : !llvm.i1
# CHECK-NEXT:     %4 = llvm.icmp "sle" %0, %arg1 : !llvm.i64
# CHECK-NEXT:     %5 = llvm.select %4, %arg1, %1 : !llvm.i1, !llvm.i64
# CHECK-NEXT:     %6 = llvm.icmp "slt" %5, %0 : !llvm.i64
# CHECK-NEXT:     llvm.cond_br %6, ^bb1, ^bb2(%2, %0, %0 : !llvm.i1, !llvm.i64, !llvm.i64)
# CHECK-NEXT:   ^bb1:  // pred: ^bb0
# CHECK-NEXT:     %7 = llvm.mlir.undef : !llvm.i64
# CHECK-NEXT:     llvm.br ^bb2(%3, %7, %7 : !llvm.i1, !llvm.i64, !llvm.i64)
# CHECK-NEXT:   ^bb2(%8: !llvm.i1, %9: !llvm.i64, %10: !llvm.i64):  // 2 preds: ^bb0, ^bb1
# CHECK-NEXT:     %11 = llvm.xor %8, %3 : !llvm.i1
# CHECK-NEXT:     llvm.cond_br %11, ^bb3(%9, %10, %1 : !llvm.i64, !llvm.i64, !llvm.i64), ^bb7(%1 : !llvm.i64)
# CHECK-NEXT:   ^bb3(%12: !llvm.i64, %13: !llvm.i64, %14: !llvm.i64):  // 2 preds: ^bb2, ^bb6
# CHECK-NEXT:     %15 = llvm.add %14, %12 : !llvm.i64
# CHECK-NEXT:     %16 = llvm.icmp "eq" %13, %5 : !llvm.i64
# CHECK-NEXT:     llvm.cond_br %16, ^bb4, ^bb5
# CHECK-NEXT:   ^bb4:  // pred: ^bb3
# CHECK-NEXT:     %17 = llvm.mlir.undef : !llvm.i64
# CHECK-NEXT:     llvm.br ^bb6(%17, %17, %3 : !llvm.i64, !llvm.i64, !llvm.i1)
# CHECK-NEXT:   ^bb5:  // pred: ^bb3
# CHECK-NEXT:     %18 = llvm.add %13, %0 : !llvm.i64
# CHECK-NEXT:     llvm.br ^bb6(%18, %18, %2 : !llvm.i64, !llvm.i64, !llvm.i1)
# CHECK-NEXT:   ^bb6(%19: !llvm.i64, %20: !llvm.i64, %21: !llvm.i1):  // 2 preds: ^bb4, ^bb5
# CHECK-NEXT:     %22 = llvm.xor %21, %3 : !llvm.i1
# CHECK-NEXT:     llvm.cond_br %22, ^bb3(%19, %20, %15 : !llvm.i64, !llvm.i64, !llvm.i64), ^bb7(%15 : !llvm.i64)
# CHECK-NEXT:   ^bb7(%23: !llvm.i64):  // 2 preds: ^bb2, ^bb6
# CHECK-NEXT:     llvm.return %23 : !llvm.i64
# CHECK-NEXT:   }
# CHECK-NEXT: }
