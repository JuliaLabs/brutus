# RUN: julia -e "import Brutus; Brutus.lit(:emit_lowered)" --startup-file=no %s 2>&1 | FileCheck %s

select(c) = 1 + (c ? 2 : 3)
emit(select, Bool)




# CHECK: Core.MethodMatch(Tuple{typeof(Main.Main.select), Bool}, svec(), select(c) in Main.Main at /{{.*}}/test/Codegen/lower/select.jl:3, true)after translating to MLIR in JLIR dialect:module  {
# CHECK-NEXT:   func nested @"Tuple{typeof(Main.select), Bool}"(%arg0: !jlir<"typeof(Main.select)">, %arg1: !jlir.Bool) -> !jlir.Int64 attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     "jlir.goto"()[^bb1] : () -> ()
# CHECK-NEXT:   ^bb1:  // pred: ^bb0
# CHECK-NEXT:     "jlir.gotoifnot"(%arg1)[^bb3, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb2:  // pred: ^bb1
# CHECK-NEXT:     %0 = "jlir.constant"() {value = #jlir<"2">} : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%0)[^bb4] : (!jlir.Int64) -> ()
# CHECK-NEXT:   ^bb3:  // pred: ^bb1
# CHECK-NEXT:     %1 = "jlir.constant"() {value = #jlir<"3">} : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.goto"(%1)[^bb4] : (!jlir.Int64) -> ()
# CHECK-NEXT:   ^bb4(%2: !jlir.Int64):  // 2 preds: ^bb2, ^bb3
# CHECK-NEXT:     %3 = "jlir.constant"() {value = #jlir<"#<intrinsic #2 add_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %4 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK-NEXT:     %5 = "jlir.call"(%3, %4, %2) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     "jlir.return"(%5) : (!jlir.Int64) -> ()
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK: module  {
# CHECK-NEXT:   func nested @"Tuple{typeof(Main.select), Bool}"(%arg0: !jlir<"typeof(Main.select)">, %arg1: i1) -> i64 attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     %c2_i64 = constant 2 : i64
# CHECK-NEXT:     %c3_i64 = constant 3 : i64
# CHECK-NEXT:     %c1_i64 = constant 1 : i64
# CHECK-NEXT:     %0 = select %arg1, %c2_i64, %c3_i64 : i64
# CHECK-NEXT:     %1 = addi %0, %c1_i64 : i64
# CHECK-NEXT:     return %1 : i64
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK:   llvm.func @"Tuple{typeof(Main.select), Bool}"(%arg0: !llvm.ptr<struct<"struct_jl_value_type", opaque>>, %arg1: i1) -> i64 attributes {llvm.emit_c_interface, sym_visibility = "nested"} {
# CHECK-NEXT:     %0 = llvm.mlir.constant({{[0-9]+}} : i64) : i64
# CHECK-NEXT:     %1 = llvm.mlir.constant({{[0-9]+}} : i64) : i64
# CHECK-NEXT:     %2 = llvm.mlir.constant({{[0-9]+}} : i64) : i64
# CHECK-NEXT:     %3 = llvm.select %arg1, %0, %1 : i1, i64
# CHECK-NEXT:     %4 = llvm.add %3, %2  : i64
# CHECK-NEXT:     llvm.return %4 : i64
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @"_mlir_ciface_Tuple{typeof(Main.select), Bool}"(%arg0: !llvm.ptr<struct<"struct_jl_value_type", opaque>>, %arg1: i1) -> i64 attributes {llvm.emit_c_interface, sym_visibility = "nested"} {
# CHECK-NEXT:     %0 = llvm.call @"Tuple{typeof(Main.select), Bool}"(%arg0, %arg1) : (!llvm.ptr<struct<"struct_jl_value_type", opaque>>, i1) -> i64
# CHECK-NEXT:     llvm.return %0 : i64
# CHECK-NEXT:   }
# CHECK-NEXT: }
