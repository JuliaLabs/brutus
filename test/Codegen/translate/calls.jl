# RUN: julia -e "import Brutus; Brutus.lit(:emit_translated)" --startup-file=no %s 2>&1 | FileCheck %s

function calls()
    f = true ? (+) : (-)
    return f(1, 1)
end
emit(calls)


# CHECK: module  {
# CHECK-NEXT:   func nested @"Tuple{typeof(Main.calls)}"(%arg0: !jlir<"typeof(Main.calls)">, %arg1: !jlir<"typeof(Base.:(+))">, %arg2: !jlir<"typeof(Base.:(+))">) -> !jlir.Int64 attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     "jlir.goto"()[^bb1] : () -> ()
# CHECK-NEXT:   ^bb1:  // pred: ^bb0
# CHECK-NEXT:     "jlir.goto"()[^bb2] : () -> ()
# CHECK-NEXT:   ^bb2:  // pred: ^bb1
# CHECK-NEXT:     %0 = "jlir.constant"() {value = #jlir<"2">} : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.return"(%0) : (!jlir.Int64) -> ()
# CHECK-NEXT:   }
# CHECK-NEXT: }
