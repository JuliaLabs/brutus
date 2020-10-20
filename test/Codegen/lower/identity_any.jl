# RUN: julia -e "import Brutus; Brutus.lit(:emit_lowered)" --startup-file=no %s 2>&1 | FileCheck %s

emit(identity, Any)



# CHECK: module {
# CHECK-NEXT:   func @"Tuple{typeof(Base.identity), Any}"(%arg0: !jlir<"typeof(Base.identity)">, %arg1: !jlir.Any) -> !jlir.Any {
# CHECK-NEXT:     return %arg1 : !jlir.Any
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK: module {
# CHECK-NEXT:   llvm.func @"Tuple{typeof(Base.identity), Any}"(%arg0: !llvm.ptr<struct<"jl_value_t", ()>>, %arg1: !llvm.ptr<struct<"jl_value_t", ()>>) -> !llvm.ptr<struct<"jl_value_t", ()>> {
# CHECK-NEXT:     llvm.return %arg1 : !llvm.ptr<struct<"jl_value_t", ()>>
# CHECK-NEXT:   }
# CHECK-NEXT: }
