# RUN: julia -e "import Brutus; Brutus.lit(:emit_translated)" --startup-file=no %s 2>&1 | FileCheck %s

@noinline add(x, y) = x + y
function caller(x, y)
    add(x, y)
end
emit(caller, Int64, Int64)


# CHECK: module  {
# CHECK-NEXT:   func nested @"Tuple{typeof(Main.caller), Int64, Int64}"(%arg0: !jlir<"typeof(Main.caller)">, %arg1: !jlir.Int64, %arg2: !jlir.Int64) -> !jlir.Int64 attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     "jlir.goto"()[^bb1] : () -> ()
# CHECK-NEXT:   ^bb1:  // pred: ^bb0
# CHECK-NEXT:     %0 = "jlir.constant"() {value = #jlir.Main.add} : () -> !jlir<"typeof(Main.add)">
# CHECK-NEXT:     %1 = "jlir.invoke"(%0, %arg1, %arg2) {methodInstance = #jlir<"add(Int64, Int64) from add(Any, Any)">} : (!jlir<"typeof(Main.add)">, !jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK-NEXT:     "jlir.return"(%1) : (!jlir.Int64) -> ()
# CHECK-NEXT:   }
# CHECK-NEXT: }
