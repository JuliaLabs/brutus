# RUN: julia -e "import Brutus; Brutus.lit(:emit_lowered)" --startup-file=no %s 2>&1 | FileCheck %s

emit(identity, Nothing)
# CHECK: func @"Tuple{typeof(Base.identity), Nothing}"(%arg0: !jlir<"typeof(Base.identity)">, %arg1: !jlir.Nothing) -> !jlir.Nothing
# CHECK:   "jlir.return"(%arg1) : (!jlir.Nothing) -> ()
# CHECK: llvm.func @"Tuple{typeof(Base.identity), Nothing}"(%arg0: !llvm<"%jl_value_t*">, %arg1: !llvm<"%jl_value_t*">) -> !llvm<"%jl_value_t*">
# CHECK:   llvm.return %arg1 : !llvm<"%jl_value_t*">