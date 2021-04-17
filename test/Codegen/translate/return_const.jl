# RUN: julia -e "import Brutus; Brutus.lit(:emit_translated)" --startup-file=no %s 2>&1 | FileCheck %s

f() = return 2
emit(f)


# CHECK: Core.MethodMatch(Tuple{typeof(Main.Main.f)}, svec(), f() in Main.Main at /home/mccoy/Dev/brutus/test/Codegen/translate/return_const.jl:3, true)after translating to MLIR in JLIR dialect:module  {
# CHECK-NEXT:   func nested @"Tuple{typeof(Main.f)}"(%arg0: !jlir<"typeof(Main.f)">) -> !jlir.Int64 attributes {llvm.emit_c_interface} {
# CHECK-NEXT:     "jlir.goto"()[^bb1] : () -> ()
# CHECK-NEXT:   ^bb1:  // pred: ^bb0
# CHECK-NEXT:     %0 = "jlir.constant"() {value = #jlir<"2">} : () -> !jlir.Int64
# CHECK-NEXT:     "jlir.return"(%0) : (!jlir.Int64) -> ()
# CHECK-NEXT:   }
# CHECK-NEXT: }
