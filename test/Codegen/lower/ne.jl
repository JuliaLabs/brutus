# RUN: julia -e "import Brutus; Brutus.lit(:emit_lowered)" --startup-file=no %s 2>&1 | FileCheck %s

ne(x, y) = x != y
emit(ne, Float64, Float64)



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
