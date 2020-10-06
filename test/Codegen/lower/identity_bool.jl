# RUN: julia -e "import Brutus; Brutus.lit(:emit_lowered)" --startup-file=no %s 2>&1 | FileCheck %s

emit(identity, Bool)



# CHECK: module {
# CHECK-NEXT:   func @"Tuple{typeof(Base.identity), Bool}"(%arg0: !jlir<"typeof(Base.identity)">, %arg1: i1) -> i1 {
# CHECK-NEXT:     return %arg1 : i1
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK: module {
# CHECK-NEXT:   llvm.func @"Tuple{typeof(Base.identity), Bool}"(%arg0: !llvm<"%jl_value_t*">, %arg1: !llvm.i1) -> !llvm.i1 {
# CHECK-NEXT:     llvm.return %arg1 : !llvm.i1
# CHECK-NEXT:   }
# CHECK-NEXT: }
