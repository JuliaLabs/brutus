# RUN: julia -e "import Brutus; Brutus.lit(:emit_optimized)" --startup-file=no %s 2>&1 | FileCheck %s

function branches(c)
    if c
        return c
    else
        return !c
    end
end
emit(branches, Bool)



# CHECK: Core.MethodMatch(Tuple{typeof(Main.Main.branches), Bool}, svec(), branches(c) in Main.Main at /{{.*}}/test/Codegen/canonicalize/branches.jl:3, true)after translating to MLIR in JLIR dialect:module  {
# CHECK-NEXT:   func nested @"Tuple{typeof(Main.branches), Bool}"(%arg0: !jlir<"typeof(Main.branches)">, %arg1: !jlir.Bool) -> !jlir.Bool attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     "jlir.goto"()[^bb1] : () -> ()
# CHECK-NEXT:   ^bb1:  // pred: ^bb0
# CHECK-NEXT:     "jlir.gotoifnot"(%arg1)[^bb3, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb2:  // pred: ^bb1
# CHECK-NEXT:     "jlir.return"(%arg1) : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb3:  // pred: ^bb1
# CHECK-NEXT:     %0 = "jlir.constant"() {value = #jlir<"#<intrinsic #44 not_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK-NEXT:     %1 = "jlir.call"(%0, %arg1) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     "jlir.return"(%1) : (!jlir.Bool) -> ()
# CHECK-NEXT:   }
# CHECK-NEXT: }

# CHECK:   func nested @"Tuple{typeof(Main.branches), Bool}"(%arg0: !jlir<"typeof(Main.branches)">, %arg1: !jlir.Bool) -> !jlir.Bool attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     "jlir.goto"()[^bb1] : () -> ()
# CHECK-NEXT:   ^bb1:  // pred: ^bb0
# CHECK-NEXT:     "jlir.gotoifnot"(%arg1)[^bb3, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb2:  // pred: ^bb1
# CHECK-NEXT:     "jlir.return"(%arg1) : (!jlir.Bool) -> ()
# CHECK-NEXT:   ^bb3:  // pred: ^bb1
# CHECK-NEXT:     %0 = "jlir.not_int"(%arg1) : (!jlir.Bool) -> !jlir.Bool
# CHECK-NEXT:     "jlir.return"(%0) : (!jlir.Bool) -> ()
# CHECK-NEXT:   }
# CHECK-NEXT: }
