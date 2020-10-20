# RUN: julia -e "import Brutus; Brutus.lit(:emit_lowered)" --startup-file=no %s 2>&1 | FileCheck %s

add(x, y) = x + y
emit(add, Float64, Float64)



# CHECK: module {
# CHECK-NEXT:   func @"Tuple{typeof(Main.add), Float64, Float64}"(%arg0: !jlir<"typeof(Main.add)">, %arg1: f64, %arg2: f64) -> f64 {
# CHECK-NEXT:     %0 = addf %arg1, %arg2 : f64
# CHECK-NEXT:     return %0 : f64
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK: module {
# CHECK-NEXT:   llvm.func @"Tuple{typeof(Main.add), Float64, Float64}"(%arg0: !llvm.ptr<struct<"jl_value_t", ()>>, %arg1: !llvm.double, %arg2: !llvm.double) -> !llvm.double {
# CHECK-NEXT:     %0 = llvm.fadd %arg1, %arg2 : !llvm.double
# CHECK-NEXT:     llvm.return %0 : !llvm.double
# CHECK-NEXT:   }
# CHECK-NEXT: }
