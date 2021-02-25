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
# CHECK-NEXT:   func nested @"Tuple{typeof(Main.matmul!), Array{Float64, 2}, Array{Float64, 2}, Array{Float64, 2}}"(%arg0: !jlir<"typeof(Main.matmul!)">, %arg1: memref<?x?xf64>, %arg2: memref<?x?xf64>, %arg3: memref<?x?xf64>) -> !jlir.Nothing attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     %c1_i64 = constant 1 : i64
# CHECK-NEXT:     %c0_i64 = constant 0 : i64
# CHECK-NEXT:     %c2_i64 = constant 2 : i64
# CHECK-NEXT:     %c2 = constant 2 : index
# CHECK-NEXT:     %c1 = constant 1 : index
# CHECK-NEXT:     %false = constant false
# CHECK-NEXT:     %true = constant true
# CHECK-NEXT:     %0 = "jlir.convertstd"(%c1_i64) : (i64) -> index
# CHECK-NEXT:     %1 = subi %c2, %0 : index
# CHECK-NEXT:     %2 = dim %arg2, %1 : memref<?x?xf64>
# CHECK-NEXT:     %3 = "jlir.convertstd"(%2) : (index) -> i64
# CHECK-NEXT:     %4 = cmpi sle, %c1_i64, %3 : i64
# CHECK-NEXT:     %5 = select %4, %3, %c0_i64 : i64
# CHECK-NEXT:     %6 = "jlir.convertstd"(%5) : (i64) -> !jlir.Int64
# CHECK-NEXT:     %7 = cmpi slt, %5, %c1_i64 : i64
# CHECK-NEXT:     cond_br %7, ^bb1, ^bb2(%false, %c1_i64, %c1_i64 : i1, i64, i64)
# CHECK-NEXT:   ^bb1:  // pred: ^bb0
# CHECK-NEXT:     %8 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %9 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %10 = "jlir.convertstd"(%8) : (!jlir.Int64) -> i64
# CHECK-NEXT:     %11 = "jlir.convertstd"(%9) : (!jlir.Int64) -> i64
# CHECK-NEXT:     br ^bb2(%true, %10, %11 : i1, i64, i64)
# CHECK-NEXT:   ^bb2(%12: i1, %13: i64, %14: i64):  // 2 preds: ^bb0, ^bb1
# CHECK-NEXT:     %15 = xor %12, %true : i1
# CHECK-NEXT:     cond_br %15, ^bb3(%13, %14 : i64, i64), ^bb21
# CHECK-NEXT:   ^bb3(%16: i64, %17: i64):  // 2 preds: ^bb2, ^bb20
# CHECK-NEXT:     %18 = "jlir.convertstd"(%17) : (i64) -> !jlir.Int64
# CHECK-NEXT:     %19 = "jlir.convertstd"(%c2_i64) : (i64) -> index
# CHECK-NEXT:     %20 = subi %c2, %19 : index
# CHECK-NEXT:     %21 = dim %arg2, %20 : memref<?x?xf64>
# CHECK-NEXT:     %22 = "jlir.convertstd"(%21) : (index) -> i64
# CHECK-NEXT:     %23 = cmpi sle, %c1_i64, %22 : i64
# CHECK-NEXT:     %24 = select %23, %22, %c0_i64 : i64
# CHECK-NEXT:     %25 = "jlir.convertstd"(%24) : (i64) -> !jlir.Int64
# CHECK-NEXT:     %26 = cmpi slt, %24, %c1_i64 : i64
# CHECK-NEXT:     cond_br %26, ^bb4, ^bb5(%false, %c1_i64, %c1_i64 : i1, i64, i64)
# CHECK-NEXT:   ^bb4:  // pred: ^bb3
# CHECK-NEXT:     %27 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %28 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %29 = "jlir.convertstd"(%27) : (!jlir.Int64) -> i64
# CHECK-NEXT:     %30 = "jlir.convertstd"(%28) : (!jlir.Int64) -> i64
# CHECK-NEXT:     br ^bb5(%true, %29, %30 : i1, i64, i64)
# CHECK-NEXT:   ^bb5(%31: i1, %32: i64, %33: i64):  // 2 preds: ^bb3, ^bb4
# CHECK-NEXT:     %34 = xor %31, %true : i1
# CHECK-NEXT:     cond_br %34, ^bb6(%32, %33 : i64, i64), ^bb17
# CHECK-NEXT:   ^bb6(%35: i64, %36: i64):  // 2 preds: ^bb5, ^bb16
# CHECK-NEXT:     %37 = "jlir.convertstd"(%36) : (i64) -> !jlir.Int64
# CHECK-NEXT:     %38 = dim %arg3, %20 : memref<?x?xf64>
# CHECK-NEXT:     %39 = "jlir.convertstd"(%38) : (index) -> i64
# CHECK-NEXT:     %40 = cmpi sle, %c1_i64, %39 : i64
# CHECK-NEXT:     %41 = select %40, %39, %c0_i64 : i64
# CHECK-NEXT:     %42 = "jlir.convertstd"(%41) : (i64) -> !jlir.Int64
# CHECK-NEXT:     %43 = cmpi slt, %41, %c1_i64 : i64
# CHECK-NEXT:     cond_br %43, ^bb7, ^bb8(%false, %c1_i64, %c1_i64 : i1, i64, i64)
# CHECK-NEXT:   ^bb7:  // pred: ^bb6
# CHECK-NEXT:     %44 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %45 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %46 = "jlir.convertstd"(%44) : (!jlir.Int64) -> i64
# CHECK-NEXT:     %47 = "jlir.convertstd"(%45) : (!jlir.Int64) -> i64
# CHECK-NEXT:     br ^bb8(%true, %46, %47 : i1, i64, i64)
# CHECK-NEXT:   ^bb8(%48: i1, %49: i64, %50: i64):  // 2 preds: ^bb6, ^bb7
# CHECK-NEXT:     %51 = xor %48, %true : i1
# CHECK-NEXT:     cond_br %51, ^bb9(%49, %50 : i64, i64), ^bb13
# CHECK-NEXT:   ^bb9(%52: i64, %53: i64):  // 2 preds: ^bb8, ^bb12
# CHECK-NEXT:     %54 = "jlir.convertstd"(%53) : (i64) -> !jlir.Int64
# CHECK-NEXT:     %55 = "jlir.convertstd"(%16) : (i64) -> index
# CHECK-NEXT:     %56 = subi %55, %c1 : index
# CHECK-NEXT:     %57 = "jlir.convertstd"(%52) : (i64) -> index
# CHECK-NEXT:     %58 = subi %57, %c1 : index
# CHECK-NEXT:     %59 = load %arg1[%58, %56] : memref<?x?xf64>
# CHECK-NEXT:     %60 = "jlir.convertstd"(%35) : (i64) -> index
# CHECK-NEXT:     %61 = subi %60, %c1 : index
# CHECK-NEXT:     %62 = load %arg2[%61, %56] : memref<?x?xf64>
# CHECK-NEXT:     %63 = load %arg3[%58, %61] : memref<?x?xf64>
# CHECK-NEXT:     %64 = mulf %62, %63 : f64
# CHECK-NEXT:     %65 = addf %59, %64 : f64
# CHECK-NEXT:     store %65, %arg1[%58, %56] : memref<?x?xf64>
# CHECK-NEXT:     %66 = "jlir.==="(%54, %42) : (!jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     %67 = "jlir.convertstd"(%66) : (!jlir.Bool) -> i1
# CHECK-NEXT:     cond_br %67, ^bb10, ^bb11
# CHECK-NEXT:   ^bb10:  // pred: ^bb9
# CHECK-NEXT:     %68 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %69 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %70 = "jlir.convertstd"(%68) : (!jlir.Int64) -> i64
# CHECK-NEXT:     %71 = "jlir.convertstd"(%69) : (!jlir.Int64) -> i64
# CHECK-NEXT:     br ^bb12(%70, %71, %true : i64, i64, i1)
# CHECK-NEXT:   ^bb11:  // pred: ^bb9
# CHECK-NEXT:     %72 = addi %53, %c1_i64 : i64
# CHECK-NEXT:     br ^bb12(%72, %72, %false : i64, i64, i1)
# CHECK-NEXT:   ^bb12(%73: i64, %74: i64, %75: i1):  // 2 preds: ^bb10, ^bb11
# CHECK-NEXT:     %76 = xor %75, %true : i1
# CHECK-NEXT:     cond_br %76, ^bb9(%73, %74 : i64, i64), ^bb13
# CHECK-NEXT:   ^bb13:  // 2 preds: ^bb8, ^bb12
# CHECK-NEXT:     %77 = "jlir.==="(%37, %25) : (!jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     %78 = "jlir.convertstd"(%77) : (!jlir.Bool) -> i1
# CHECK-NEXT:     cond_br %78, ^bb14, ^bb15
# CHECK-NEXT:   ^bb14:  // pred: ^bb13
# CHECK-NEXT:     %79 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %80 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %81 = "jlir.convertstd"(%79) : (!jlir.Int64) -> i64
# CHECK-NEXT:     %82 = "jlir.convertstd"(%80) : (!jlir.Int64) -> i64
# CHECK-NEXT:     br ^bb16(%81, %82, %true : i64, i64, i1)
# CHECK-NEXT:   ^bb15:  // pred: ^bb13
# CHECK-NEXT:     %83 = addi %36, %c1_i64 : i64
# CHECK-NEXT:     br ^bb16(%83, %83, %false : i64, i64, i1)
# CHECK-NEXT:   ^bb16(%84: i64, %85: i64, %86: i1):  // 2 preds: ^bb14, ^bb15
# CHECK-NEXT:     %87 = xor %86, %true : i1
# CHECK-NEXT:     cond_br %87, ^bb6(%84, %85 : i64, i64), ^bb17
# CHECK-NEXT:   ^bb17:  // 2 preds: ^bb5, ^bb16
# CHECK-NEXT:     %88 = "jlir.==="(%18, %6) : (!jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK-NEXT:     %89 = "jlir.convertstd"(%88) : (!jlir.Bool) -> i1
# CHECK-NEXT:     cond_br %89, ^bb18, ^bb19
# CHECK-NEXT:   ^bb18:  // pred: ^bb17
# CHECK-NEXT:     %90 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %91 = "jlir.undef"() : () -> !jlir.Int64
# CHECK-NEXT:     %92 = "jlir.convertstd"(%90) : (!jlir.Int64) -> i64
# CHECK-NEXT:     %93 = "jlir.convertstd"(%91) : (!jlir.Int64) -> i64
# CHECK-NEXT:     br ^bb20(%92, %93, %true : i64, i64, i1)
# CHECK-NEXT:   ^bb19:  // pred: ^bb17
# CHECK-NEXT:     %94 = addi %17, %c1_i64 : i64
# CHECK-NEXT:     br ^bb20(%94, %94, %false : i64, i64, i1)
# CHECK-NEXT:   ^bb20(%95: i64, %96: i64, %97: i1):  // 2 preds: ^bb18, ^bb19
# CHECK-NEXT:     %98 = xor %97, %true : i1
# CHECK-NEXT:     cond_br %98, ^bb3(%95, %96 : i64, i64), ^bb21
# CHECK-NEXT:   ^bb21:  // 2 preds: ^bb2, ^bb20
# CHECK-NEXT:     %99 = "jlir.constant"() {value = #jlir.nothing} : () -> !jlir.Nothing
# CHECK-NEXT:     return %99 : !jlir.Nothing
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK:   llvm.func @"Tuple{typeof(Main.matmul!), Array{Float64, 2}, Array{Float64, 2}, Array{Float64, 2}}"(%arg0: !llvm.ptr<struct<"struct_jl_value_type", opaque>>, %arg1: !llvm.ptr<f64>, %arg2: !llvm.ptr<f64>, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: !llvm.ptr<f64>, %arg9: !llvm.ptr<f64>, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: !llvm.ptr<f64>, %arg16: !llvm.ptr<f64>, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64, %arg21: i64) -> !llvm.ptr<struct<"struct_jl_value_type", opaque>> attributes {llvm.emit_c_interface, sym_visibility = "nested"} {
# CHECK-NEXT:     %0 = llvm.mlir.undef : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %1 = llvm.insertvalue %arg1, %0[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %2 = llvm.insertvalue %arg2, %1[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %3 = llvm.insertvalue %arg3, %2[2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %4 = llvm.insertvalue %arg4, %3[3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %5 = llvm.insertvalue %arg6, %4[4, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %6 = llvm.insertvalue %arg5, %5[3, 1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %7 = llvm.insertvalue %arg7, %6[4, 1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %8 = llvm.insertvalue %arg8, %0[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %9 = llvm.insertvalue %arg9, %8[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %10 = llvm.insertvalue %arg10, %9[2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %11 = llvm.insertvalue %arg11, %10[3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %12 = llvm.insertvalue %arg13, %11[4, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %13 = llvm.insertvalue %arg12, %12[3, 1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %14 = llvm.insertvalue %arg14, %13[4, 1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %15 = llvm.insertvalue %arg15, %0[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %16 = llvm.insertvalue %arg16, %15[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %17 = llvm.insertvalue %arg17, %16[2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %18 = llvm.insertvalue %arg18, %17[3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %19 = llvm.insertvalue %arg20, %18[4, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %20 = llvm.insertvalue %arg19, %19[3, 1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %21 = llvm.insertvalue %arg21, %20[4, 1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %22 = llvm.mlir.constant({{[0-9]+}} : i64) : i64
# CHECK-NEXT:     %23 = llvm.mlir.constant({{[0-9]+}} : i64) : i64
# CHECK-NEXT:     %24 = llvm.mlir.constant({{[0-9]+}} : i64) : i64
# CHECK-NEXT:     %25 = llvm.mlir.constant(2 : index) : i64
# CHECK-NEXT:     %26 = llvm.mlir.constant(1 : index) : i64
# CHECK-NEXT:     %27 = llvm.mlir.constant(false) : i1
# CHECK-NEXT:     %28 = llvm.mlir.constant(true) : i1
# CHECK-NEXT:     %29 = llvm.sub %25, %22  : i64
# CHECK-NEXT:     %30 = llvm.mlir.constant(0 : index) : i64
# CHECK-NEXT:     %31 = llvm.extractvalue %14[3] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %32 = llvm.alloca %26 x !llvm.array<2 x i64> : (i64) -> !llvm.ptr<array<2 x i64>>
# CHECK-NEXT:     llvm.store %31, %32 : !llvm.ptr<array<2 x i64>>
# CHECK-NEXT:     %33 = llvm.getelementptr %32[%30, %29] : (!llvm.ptr<array<2 x i64>>, i64, i64) -> !llvm.ptr<i64>
# CHECK-NEXT:     %34 = llvm.load %33 : !llvm.ptr<i64>
# CHECK-NEXT:     %35 = llvm.icmp "sle" %22, %34 : i64
# CHECK-NEXT:     %36 = llvm.select %35, %34, %23 : i1, i64
# CHECK-NEXT:     %37 = llvm.icmp "slt" %36, %22 : i64
# CHECK-NEXT:     llvm.cond_br %37, ^bb1, ^bb2(%27, %22, %22 : i1, i64, i64)
# CHECK-NEXT:   ^bb1:  // pred: ^bb0
# CHECK-NEXT:     %38 = llvm.mlir.undef : i64
# CHECK-NEXT:     llvm.br ^bb2(%28, %38, %38 : i1, i64, i64)
# CHECK-NEXT:   ^bb2(%39: i1, %40: i64, %41: i64):  // 2 preds: ^bb0, ^bb1
# CHECK-NEXT:     %42 = llvm.xor %39, %28  : i1
# CHECK-NEXT:     llvm.cond_br %42, ^bb3(%40, %41 : i64, i64), ^bb21
# CHECK-NEXT:   ^bb3(%43: i64, %44: i64):  // 2 preds: ^bb2, ^bb20
# CHECK-NEXT:     %45 = llvm.sub %25, %24  : i64
# CHECK-NEXT:     %46 = llvm.alloca %26 x !llvm.array<2 x i64> : (i64) -> !llvm.ptr<array<2 x i64>>
# CHECK-NEXT:     llvm.store %31, %46 : !llvm.ptr<array<2 x i64>>
# CHECK-NEXT:     %47 = llvm.getelementptr %46[%30, %45] : (!llvm.ptr<array<2 x i64>>, i64, i64) -> !llvm.ptr<i64>
# CHECK-NEXT:     %48 = llvm.load %47 : !llvm.ptr<i64>
# CHECK-NEXT:     %49 = llvm.icmp "sle" %22, %48 : i64
# CHECK-NEXT:     %50 = llvm.select %49, %48, %23 : i1, i64
# CHECK-NEXT:     %51 = llvm.icmp "slt" %50, %22 : i64
# CHECK-NEXT:     llvm.cond_br %51, ^bb4, ^bb5(%27, %22, %22 : i1, i64, i64)
# CHECK-NEXT:   ^bb4:  // pred: ^bb3
# CHECK-NEXT:     %52 = llvm.mlir.undef : i64
# CHECK-NEXT:     llvm.br ^bb5(%28, %52, %52 : i1, i64, i64)
# CHECK-NEXT:   ^bb5(%53: i1, %54: i64, %55: i64):  // 2 preds: ^bb3, ^bb4
# CHECK-NEXT:     %56 = llvm.xor %53, %28  : i1
# CHECK-NEXT:     llvm.cond_br %56, ^bb6(%54, %55 : i64, i64), ^bb17
# CHECK-NEXT:   ^bb6(%57: i64, %58: i64):  // 2 preds: ^bb5, ^bb16
# CHECK-NEXT:     %59 = llvm.extractvalue %21[3] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %60 = llvm.alloca %26 x !llvm.array<2 x i64> : (i64) -> !llvm.ptr<array<2 x i64>>
# CHECK-NEXT:     llvm.store %59, %60 : !llvm.ptr<array<2 x i64>>
# CHECK-NEXT:     %61 = llvm.getelementptr %60[%30, %45] : (!llvm.ptr<array<2 x i64>>, i64, i64) -> !llvm.ptr<i64>
# CHECK-NEXT:     %62 = llvm.load %61 : !llvm.ptr<i64>
# CHECK-NEXT:     %63 = llvm.icmp "sle" %22, %62 : i64
# CHECK-NEXT:     %64 = llvm.select %63, %62, %23 : i1, i64
# CHECK-NEXT:     %65 = llvm.icmp "slt" %64, %22 : i64
# CHECK-NEXT:     llvm.cond_br %65, ^bb7, ^bb8(%27, %22, %22 : i1, i64, i64)
# CHECK-NEXT:   ^bb7:  // pred: ^bb6
# CHECK-NEXT:     %66 = llvm.mlir.undef : i64
# CHECK-NEXT:     llvm.br ^bb8(%28, %66, %66 : i1, i64, i64)
# CHECK-NEXT:   ^bb8(%67: i1, %68: i64, %69: i64):  // 2 preds: ^bb6, ^bb7
# CHECK-NEXT:     %70 = llvm.xor %67, %28  : i1
# CHECK-NEXT:     llvm.cond_br %70, ^bb9(%68, %69 : i64, i64), ^bb13
# CHECK-NEXT:   ^bb9(%71: i64, %72: i64):  // 2 preds: ^bb8, ^bb12
# CHECK-NEXT:     %73 = llvm.sub %43, %26  : i64
# CHECK-NEXT:     %74 = llvm.sub %71, %26  : i64
# CHECK-NEXT:     %75 = llvm.extractvalue %7[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %76 = llvm.extractvalue %7[4, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %77 = llvm.mul %74, %76  : i64
# CHECK-NEXT:     %78 = llvm.add %77, %73  : i64
# CHECK-NEXT:     %79 = llvm.getelementptr %75[%78] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
# CHECK-NEXT:     %80 = llvm.load %79 : !llvm.ptr<f64>
# CHECK-NEXT:     %81 = llvm.sub %57, %26  : i64
# CHECK-NEXT:     %82 = llvm.extractvalue %14[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %83 = llvm.extractvalue %14[4, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %84 = llvm.mul %81, %83  : i64
# CHECK-NEXT:     %85 = llvm.add %84, %73  : i64
# CHECK-NEXT:     %86 = llvm.getelementptr %82[%85] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
# CHECK-NEXT:     %87 = llvm.load %86 : !llvm.ptr<f64>
# CHECK-NEXT:     %88 = llvm.extractvalue %21[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %89 = llvm.extractvalue %21[4, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
# CHECK-NEXT:     %90 = llvm.mul %74, %89  : i64
# CHECK-NEXT:     %91 = llvm.add %90, %81  : i64
# CHECK-NEXT:     %92 = llvm.getelementptr %88[%91] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
# CHECK-NEXT:     %93 = llvm.load %92 : !llvm.ptr<f64>
# CHECK-NEXT:     %94 = llvm.fmul %87, %93  : f64
# CHECK-NEXT:     %95 = llvm.fadd %80, %94  : f64
# CHECK-NEXT:     llvm.store %95, %79 : !llvm.ptr<f64>
# CHECK-NEXT:     %96 = llvm.icmp "eq" %72, %64 : i64
# CHECK-NEXT:     llvm.cond_br %96, ^bb10, ^bb11
# CHECK-NEXT:   ^bb10:  // pred: ^bb9
# CHECK-NEXT:     %97 = llvm.mlir.undef : i64
# CHECK-NEXT:     llvm.br ^bb12(%97, %97, %28 : i64, i64, i1)
# CHECK-NEXT:   ^bb11:  // pred: ^bb9
# CHECK-NEXT:     %98 = llvm.add %72, %22  : i64
# CHECK-NEXT:     llvm.br ^bb12(%98, %98, %27 : i64, i64, i1)
# CHECK-NEXT:   ^bb12(%99: i64, %100: i64, %101: i1):  // 2 preds: ^bb10, ^bb11
# CHECK-NEXT:     %102 = llvm.xor %101, %28  : i1
# CHECK-NEXT:     llvm.cond_br %102, ^bb9(%99, %100 : i64, i64), ^bb13
# CHECK-NEXT:   ^bb13:  // 2 preds: ^bb8, ^bb12
# CHECK-NEXT:     %103 = llvm.icmp "eq" %58, %50 : i64
# CHECK-NEXT:     llvm.cond_br %103, ^bb14, ^bb15
# CHECK-NEXT:   ^bb14:  // pred: ^bb13
# CHECK-NEXT:     %104 = llvm.mlir.undef : i64
# CHECK-NEXT:     llvm.br ^bb16(%104, %104, %28 : i64, i64, i1)
# CHECK-NEXT:   ^bb15:  // pred: ^bb13
# CHECK-NEXT:     %105 = llvm.add %58, %22  : i64
# CHECK-NEXT:     llvm.br ^bb16(%105, %105, %27 : i64, i64, i1)
# CHECK-NEXT:   ^bb16(%106: i64, %107: i64, %108: i1):  // 2 preds: ^bb14, ^bb15
# CHECK-NEXT:     %109 = llvm.xor %108, %28  : i1
# CHECK-NEXT:     llvm.cond_br %109, ^bb6(%106, %107 : i64, i64), ^bb17
# CHECK-NEXT:   ^bb17:  // 2 preds: ^bb5, ^bb16
# CHECK-NEXT:     %110 = llvm.icmp "eq" %44, %36 : i64
# CHECK-NEXT:     llvm.cond_br %110, ^bb18, ^bb19
# CHECK-NEXT:   ^bb18:  // pred: ^bb17
# CHECK-NEXT:     %111 = llvm.mlir.undef : i64
# CHECK-NEXT:     llvm.br ^bb20(%111, %111, %28 : i64, i64, i1)
# CHECK-NEXT:   ^bb19:  // pred: ^bb17
# CHECK-NEXT:     %112 = llvm.add %44, %22  : i64
# CHECK-NEXT:     llvm.br ^bb20(%112, %112, %27 : i64, i64, i1)
# CHECK-NEXT:   ^bb20(%113: i64, %114: i64, %115: i1):  // 2 preds: ^bb18, ^bb19
# CHECK-NEXT:     %116 = llvm.xor %115, %28  : i1
# CHECK-NEXT:     llvm.cond_br %116, ^bb3(%113, %114 : i64, i64), ^bb21
# CHECK-NEXT:   ^bb21:  // 2 preds: ^bb2, ^bb20
# CHECK-NEXT:     %117 = llvm.mlir.constant({{[0-9]+}} : i64) : i64
# CHECK-NEXT:     %118 = llvm.inttoptr %117 : i64 to !llvm.ptr<struct<"struct_jl_value_type", opaque>>
# CHECK-NEXT:     llvm.return %118 : !llvm.ptr<struct<"struct_jl_value_type", opaque>>
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @"_mlir_ciface_Tuple{typeof(Main.matmul!), Array{Float64, 2}, Array{Float64, 2}, Array{Float64, 2}}"(%arg0: !llvm.ptr<struct<"struct_jl_value_type", opaque>>, %arg1: !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>>, %arg2: !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>>, %arg3: !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>>) -> !llvm.ptr<struct<"struct_jl_value_type", opaque>> attributes {llvm.emit_c_interface, sym_visibility = "nested"} {
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
# CHECK-NEXT:     %24 = llvm.call @"Tuple{typeof(Main.matmul!), Array{Float64, 2}, Array{Float64, 2}, Array{Float64, 2}}"(%arg0, %1, %2, %3, %4, %5, %6, %7, %9, %10, %11, %12, %13, %14, %15, %17, %18, %19, %20, %21, %22, %23) : (!llvm.ptr<struct<"struct_jl_value_type", opaque>>, !llvm.ptr<f64>, !llvm.ptr<f64>, i64, i64, i64, i64, i64, !llvm.ptr<f64>, !llvm.ptr<f64>, i64, i64, i64, i64, i64, !llvm.ptr<f64>, !llvm.ptr<f64>, i64, i64, i64, i64, i64) -> !llvm.ptr<struct<"struct_jl_value_type", opaque>>
# CHECK-NEXT:     llvm.return %24 : !llvm.ptr<struct<"struct_jl_value_type", opaque>>
# CHECK-NEXT:   }
# CHECK-NEXT: }
