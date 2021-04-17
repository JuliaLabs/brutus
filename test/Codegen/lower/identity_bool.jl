# RUN: julia -e "import Brutus; Brutus.lit(:emit_lowered)" --startup-file=no %s 2>&1 | FileCheck %s

emit(identity, Bool)



# CHECK:   func nested @"Tuple{typeof(Base.identity), Bool}"(%arg0: !jlir<"typeof(Base.identity)">, %arg1: !jlir.Bool) -> !jlir.Bool attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     "jlir.goto"()[^bb1] : () -> ()
# CHECK-NEXT:   ^bb1:  // pred: ^bb0
# CHECK-NEXT:     "jlir.return"(%arg1) : (!jlir.Bool) -> ()
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK: module  {
# CHECK-NEXT:   func nested @"Tuple{typeof(Base.identity), Bool}"(%arg0: !jlir<"typeof(Base.identity)">, %arg1: i1) -> i1 attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     return %arg1 : i1
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK:   llvm.func @"Tuple{typeof(Base.identity), Bool}"(%arg0: !llvm.ptr<struct<"struct_jl_value_type", opaque>>, %arg1: i1) -> i1 attributes {llvm.emit_c_interface, sym_visibility = "nested"} {
# CHECK-NEXT:     llvm.return %arg1 : i1
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @"_mlir_ciface_Tuple{typeof(Base.identity), Bool}"(%arg0: !llvm.ptr<struct<"struct_jl_value_type", opaque>>, %arg1: i1) -> i1 attributes {llvm.emit_c_interface, sym_visibility = "nested"} {
# CHECK-NEXT:     %0 = llvm.call @"Tuple{typeof(Base.identity), Bool}"(%arg0, %arg1) : (!llvm.ptr<struct<"struct_jl_value_type", opaque>>, i1) -> i1
# CHECK-NEXT:     llvm.return %0 : i1
# CHECK-NEXT:   }
# CHECK-NEXT: }
