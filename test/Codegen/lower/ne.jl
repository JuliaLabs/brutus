# RUN: julia -e "import Brutus; Brutus.lit(:emit_lowered)" --startup-file=no %s 2>&1 | FileCheck %s

ne(x, y) = x != y
emit(ne, Float64, Float64)




# CHECK: Core.MethodMatch(Tuple{typeof(Main.Main.ne), Float64, Float64}, svec(), ne(x, y) in Main.Main at /home/mccoy/Dev/brutus/test/Codegen/lower/ne.jl:3, true)after translating to MLIR in JLIR dialect:module  {
# CHECK-NEXT:   func nested @"Tuple{typeof(Main.ne), Float64, Float64}"(%arg0: !jlir<"typeof(Main.ne)">, %arg1: !jlir.Float64, %arg2: !jlir.Float64) -> !jlir.Bool attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     "jlir.goto"()[^bb1] : () -> ()
# CHECK-NEXT:   ^bb1:  // pred: ^bb0
# CHECK-NEXT:     %0 = "jlir.constant"() {value = #jlir<"#<intrinsic #32 ne_float>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %1 = "jlir.call"(%0, %arg1, %arg2) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Float64, !jlir.Float64) -> !jlir.Bool
# CHECK-NEXT:     "jlir.return"(%1) : (!jlir.Bool) -> ()
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK: module  {
# CHECK-NEXT:   func nested @"Tuple{typeof(Main.ne), Float64, Float64}"(%arg0: !jlir<"typeof(Main.ne)">, %arg1: f64, %arg2: f64) -> i1 attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     %0 = cmpf une, %arg1, %arg2 : f64
# CHECK-NEXT:     return %0 : i1
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK:   llvm.func @"Tuple{typeof(Main.ne), Float64, Float64}"(%arg0: !llvm.ptr<struct<"struct_jl_value_type", opaque>>, %arg1: f64, %arg2: f64) -> i1 attributes {llvm.emit_c_interface, sym_visibility = "nested"} {
# CHECK-NEXT:     %0 = llvm.fcmp "une" %arg1, %arg2 : f64
# CHECK-NEXT:     llvm.return %0 : i1
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @"_mlir_ciface_Tuple{typeof(Main.ne), Float64, Float64}"(%arg0: !llvm.ptr<struct<"struct_jl_value_type", opaque>>, %arg1: f64, %arg2: f64) -> i1 attributes {llvm.emit_c_interface, sym_visibility = "nested"} {
# CHECK-NEXT:     %0 = llvm.call @"Tuple{typeof(Main.ne), Float64, Float64}"(%arg0, %arg1, %arg2) : (!llvm.ptr<struct<"struct_jl_value_type", opaque>>, f64, f64) -> i1
# CHECK-NEXT:     llvm.return %0 : i1
# CHECK-NEXT:   }
# CHECK-NEXT: }
