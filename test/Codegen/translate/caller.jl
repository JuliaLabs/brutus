# RUN: julia -e "import Brutus; Brutus.lit(:emit_translated)" --startup-file=no %s 2>&1 | FileCheck %s



@noinline add(x, y) = x + y
function caller(x, y)
    add(x, y)
end
emit(caller, Int64, Int64)

# CHECK: func @"Tuple{typeof(Main.caller), Int64, Int64}"(%arg0: !jlir<"typeof(Main.caller)">, %arg1: !jlir.Int64, %arg2: !jlir.Int64) -> !jlir.Int64 {
# CHECK:   "jlir.goto"()[^bb1] : () -> ()
# CHECK: ^bb1:  // pred: ^bb0
# CHECK:   %0 = "jlir.constant"() {value = #jlir<"typeof(Main.add)()">} : () -> !jlir<"typeof(Main.add)">
# CHECK:   %1 = "jlir.invoke"(%0, %arg1, %arg2) {methodInstance = #jlir<"add(Int64, Int64)">} : (!jlir<"typeof(Main.add)">, !jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK:   "jlir.return"(%1) : (!jlir.Int64) -> ()
