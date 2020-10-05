# RUN: julia -e "import Brutus; Brutus.lit(:emit_translated)" --startup-file=no %s 2>&1 | FileCheck %s



function labels(N)
    @label start
    N += 1
    if N < 0
        @goto start
    end
    return N
end
###
# 1 ─      nothing::Nothing
# 3 2 ┄ %2 = φ (#1 => _2, #3 => %3)::Int64
#   │   %3 = Base.add_int(%2, 1)::Int64
# 4 │   %4 = Base.slt_int(%3, 0)::Bool
#   └──      goto #4 if not %4
# 5 3 ─      goto #2
# 7 4 ─      return %3
###
emit(labels, Int64)
# CHECK: func @"Tuple{typeof(Main.labels), Int64}"(%arg0: !jlir<"typeof(Main.labels)">, %arg1: !jlir.Int64) -> !jlir.Int64
# CHECK:   "jlir.goto"()[^bb1] : () -> ()
# CHECK: ^bb1:
# CHECK:   "jlir.goto"(%arg1)[^bb2] : (!jlir.Int64) -> ()
# CHECK: ^bb2(%0: !jlir.Int64):
# CHECK:   %1 = "jlir.constant"() {value = #jlir<"#<intrinsic #2 add_int>">} : () -> !jlir.Core.IntrinsicFunction
# CHECK:   %2 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK:   %3 = "jlir.call"(%1, %0, %2) : (!jlir.Core.IntrinsicFunction, !jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK:   %4 = "jlir.constant"() {value = #jlir<"#<intrinsic #27 slt_int>">} : () -> !jlir.Core.IntrinsicFunction
# CHECK:   %5 = "jlir.constant"() {value = #jlir<"0">} : () -> !jlir.Int64
# CHECK:   %6 = "jlir.call"(%4, %3, %5) : (!jlir.Core.IntrinsicFunction, !jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK:   "jlir.gotoifnot"(%6)[^bb4, ^bb3] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK: ^bb3:
# CHECK:   "jlir.goto"(%3)[^bb2] : (!jlir.Int64) -> ()
# CHECK: ^bb4:
# CHECK:   "jlir.return"(%3) : (!jlir.Int64) -> ()
