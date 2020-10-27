# RUN: julia -e "import Brutus; Brutus.lit(:emit_lowered)" --startup-file=no %s 2>&1 | FileCheck %s

emit(identity, Nothing)



# CHECK: module {
# CHECK-NEXT:   func @"Tuple{typeof(Base.identity), Nothing}"(%arg0: !jlir<"typeof(Base.identity)">, %arg1: !jlir.Nothing) -> !jlir.Nothing attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     return %arg1 : !jlir.Nothing
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK: module {
# CHECK-NEXT:   llvm.func @"Tuple{typeof(Base.identity), Nothing}"(%arg0: !llvm.ptr<struct<()>>, %arg1: !llvm.ptr<struct<()>>) -> !llvm.ptr<struct<()>> attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     llvm.return %arg1 : !llvm.ptr<struct<()>>
# CHECK-NEXT:   }
# CHECK-NEXT:   llvm.func @"_mlir_ciface_Tuple{typeof(Base.identity), Nothing}"(%arg0: !llvm.ptr<struct<()>>, %arg1: !llvm.ptr<struct<()>>) -> !llvm.ptr<struct<()>> attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     %0 = llvm.call @"Tuple{typeof(Base.identity), Nothing}"(%arg0, %arg1) : (!llvm.ptr<struct<()>>, !llvm.ptr<struct<()>>) -> !llvm.ptr<struct<()>>
# CHECK-NEXT:     llvm.return %0 : !llvm.ptr<struct<()>>
# CHECK-NEXT:   }
# CHECK-NEXT: }
