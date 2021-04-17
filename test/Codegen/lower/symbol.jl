# RUN: julia -e "import Brutus; Brutus.lit(:emit_lowered)" --startup-file=no %s 2>&1 | FileCheck %s

symbol() = :testing
emit(symbol)



# CHECK:   func nested @"Tuple{typeof(Main.symbol)}"(%arg0: !jlir<"typeof(Main.symbol)">) -> !jlir.Symbol attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     "jlir.goto"()[^bb1] : () -> ()
# CHECK-NEXT:   ^bb1:  // pred: ^bb0
# CHECK-NEXT:     %0 = "jlir.constant"() {value = #jlir<":testing">} : () -> !jlir.Symbol
# CHECK-NEXT:     "jlir.return"(%0) : (!jlir.Symbol) -> ()
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK: module  {
# CHECK-NEXT:   func nested @"Tuple{typeof(Main.symbol)}"(%arg0: !jlir<"typeof(Main.symbol)">) -> !jlir.Symbol attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     %0 = "jlir.constant"() {value = #jlir<":testing">} : () -> !jlir.Symbol
# CHECK-NEXT:     return %0 : !jlir.Symbol
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK:   llvm.func @"Tuple{typeof(Main.symbol)}"(%arg0: !llvm.ptr<struct<"struct_jl_value_type", opaque>>) -> !llvm.ptr<struct<"struct_jl_value_type", opaque>> attributes {llvm.emit_c_interface, sym_visibility = "nested"} {
# CHECK-NEXT:     %0 = llvm.mlir.constant({{[0-9]+}} : i64) : i64
# CHECK-NEXT:     %1 = llvm.inttoptr %0 : i64 to !llvm.ptr<struct<"struct_jl_value_type", opaque>>
# CHECK-NEXT:     llvm.return %1 : !llvm.ptr<struct<"struct_jl_value_type", opaque>>
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @"_mlir_ciface_Tuple{typeof(Main.symbol)}"(%arg0: !llvm.ptr<struct<"struct_jl_value_type", opaque>>) -> !llvm.ptr<struct<"struct_jl_value_type", opaque>> attributes {llvm.emit_c_interface, sym_visibility = "nested"} {
# CHECK-NEXT:     %0 = llvm.call @"Tuple{typeof(Main.symbol)}"(%arg0) : (!llvm.ptr<struct<"struct_jl_value_type", opaque>>) -> !llvm.ptr<struct<"struct_jl_value_type", opaque>>
# CHECK-NEXT:     llvm.return %0 : !llvm.ptr<struct<"struct_jl_value_type", opaque>>
# CHECK-NEXT:   }
# CHECK-NEXT: }
