# RUN: julia -e "import Brutus; Brutus.lit(:emit_lowered)" --startup-file=no %s 2>&1 | FileCheck %s

emit(identity, Nothing)



# CHECK: module {
# CHECK-NEXT:   func @"Tuple{typeof(Base.identity), Nothing}"(%arg0: !jlir<"typeof(Base.identity)">, %arg1: !jlir.Nothing) -> !jlir.Nothing {
# CHECK-NEXT:     return %arg1 : !jlir.Nothing
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK: module {
# CHECK-NEXT:   llvm.func @"Tuple{typeof(Base.identity), Nothing}"(%arg0: !llvm.ptr<struct<"jl_value_t", ()>>, %arg1: !llvm.ptr<struct<"jl_value_t", ()>>) -> !llvm.ptr<struct<"jl_value_t", ()>> {
# CHECK-NEXT:     llvm.return %arg1 : !llvm.ptr<struct<"jl_value_t", ()>>
# CHECK-NEXT:   }
# CHECK-NEXT: }
