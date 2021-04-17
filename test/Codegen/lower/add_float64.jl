# RUN: julia -e "import Brutus; Brutus.lit(:emit_lowered)" --startup-file=no %s 2>&1 | FileCheck %s

add(x, y) = x + y
emit(add, Float64, Float64)




# CHECK: Core.MethodMatch(Tuple{typeof(Main.Main.add), Float64, Float64}, svec(), add(x, y) in Main.Main at /home/mccoy/Dev/brutus/test/Codegen/lower/add_float64.jl:3, true)after translating to MLIR in JLIR dialect:module  {
# CHECK-NEXT:   func nested @"Tuple{typeof(Main.add), Float64, Float64}"(%arg0: !jlir<"typeof(Main.add)">, %arg1: !jlir.Float64, %arg2: !jlir.Float64) -> !jlir.Float64 attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     "jlir.goto"()[^bb1] : () -> ()
# CHECK-NEXT:   ^bb1:  // pred: ^bb0
# CHECK-NEXT:     %0 = "jlir.constant"() {value = #jlir<"#<intrinsic #12 add_float>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %1 = "jlir.call"(%0, %arg1, %arg2) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Float64, !jlir.Float64) -> !jlir.Float64
# CHECK-NEXT:     "jlir.return"(%1) : (!jlir.Float64) -> ()
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK: module  {
# CHECK-NEXT:   func nested @"Tuple{typeof(Main.add), Float64, Float64}"(%arg0: !jlir<"typeof(Main.add)">, %arg1: f64, %arg2: f64) -> f64 attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     %0 = addf %arg1, %arg2 : f64
# CHECK-NEXT:     return %0 : f64
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK:   llvm.func @"Tuple{typeof(Main.add), Float64, Float64}"(%arg0: !llvm.ptr<struct<"struct_jl_value_type", opaque>>, %arg1: f64, %arg2: f64) -> f64 attributes {llvm.emit_c_interface, sym_visibility = "nested"} {
# CHECK-NEXT:     %0 = llvm.fadd %arg1, %arg2  : f64
# CHECK-NEXT:     llvm.return %0 : f64
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @"_mlir_ciface_Tuple{typeof(Main.add), Float64, Float64}"(%arg0: !llvm.ptr<struct<"struct_jl_value_type", opaque>>, %arg1: f64, %arg2: f64) -> f64 attributes {llvm.emit_c_interface, sym_visibility = "nested"} {
# CHECK-NEXT:     %0 = llvm.call @"Tuple{typeof(Main.add), Float64, Float64}"(%arg0, %arg1, %arg2) : (!llvm.ptr<struct<"struct_jl_value_type", opaque>>, f64, f64) -> f64
# CHECK-NEXT:     llvm.return %0 : f64
# CHECK-NEXT:   }
# CHECK-NEXT: }
