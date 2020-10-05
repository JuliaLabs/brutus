# RUN: julia -e "import Brutus; Brutus.lit(:emit_translated)" --startup-file=no %s 2>&1 | FileCheck %s



function branches(c)
    if c
        return c
    else
        return !c
    end
end
emit(branches, Bool)
# CHECK: func @"Tuple{typeof(Main.branches), Bool}"(%arg0: !jlir<"typeof(Main.branches)">, %arg1: !jlir.Bool) -> !jlir.Bool
# CHECK:   "jlir.goto"()[^bb1] : () -> ()
# CHECK: ^bb1:
# CHECK:   "jlir.gotoifnot"(%arg1)[^bb3, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK: ^bb2:
# CHECK:   "jlir.return"(%arg1) : (!jlir.Bool) -> ()
# CHECK: ^bb3:
# CHECK:   %0 = "jlir.constant"() {value = #jlir<"#<intrinsic #44 not_int>">} : () -> !jlir.Core.IntrinsicFunction
# CHECK:   %1 = "jlir.call"(%0, %arg1) : (!jlir.Core.IntrinsicFunction, !jlir.Bool) -> !jlir.Bool
# CHECK:   "jlir.return"(%1) : (!jlir.Bool) -> ()
