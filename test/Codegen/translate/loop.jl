# RUN: julia -e "import Brutus; Brutus.lit(:emit_translated)" --startup-file=no %s 2>&1 | FileCheck %s

function loop(N)
    acc = 1
    for i in 1:N
        acc += i
    end
    return acc
end
emit(loop, Int64)

# CHECK-LABEL:   func nested @"Tuple{typeof(Main.loop), Int64}"(
# CHECK-SAME:                                                   %[[VAL_0:.*]]: !jlir<"typeof(Main.loop)">,
# CHECK-SAME:                                                   %[[VAL_1:.*]]: !jlir.Int64) -> !jlir.Int64 attributes {llvm.emit_c_interface} {
# CHECK:           "jlir.goto"()[^bb1] : () -> ()
# CHECK:         ^bb1:
# CHECK:           %[[VAL_2:.*]] = "jlir.constant"() {value = #jlir<"#<intrinsic #29 sle_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK:           %[[VAL_3:.*]] = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK:           %[[VAL_4:.*]] = "jlir.call"(%[[VAL_2]], %[[VAL_3]], %[[VAL_1]]) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK:           %[[VAL_5:.*]] = "jlir.constant"() {value = #jlir.ifelse} : () -> !jlir<"typeof(ifelse)">
# CHECK:           %[[VAL_6:.*]] = "jlir.constant"() {value = #jlir<"0">} : () -> !jlir.Int64
# CHECK:           %[[VAL_7:.*]] = "jlir.call"(%[[VAL_5]], %[[VAL_4]], %[[VAL_1]], %[[VAL_6]]) : (!jlir<"typeof(ifelse)">, !jlir.Bool, !jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK:           %[[VAL_8:.*]] = "jlir.constant"() {value = #jlir<"#<intrinsic #27 slt_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK:           %[[VAL_9:.*]] = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK:           %[[VAL_10:.*]] = "jlir.call"(%[[VAL_8]], %[[VAL_7]], %[[VAL_9]]) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK:           "jlir.gotoifnot"(%[[VAL_10]])[^bb3, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK:         ^bb2:
# CHECK:           %[[VAL_11:.*]] = "jlir.constant"() {value = #jlir.nothing} : () -> !jlir.Nothing
# CHECK:           %[[VAL_12:.*]] = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK:           %[[VAL_13:.*]] = "jlir.undef"() : () -> !jlir.Int64
# CHECK:           %[[VAL_14:.*]] = "jlir.undef"() : () -> !jlir.Int64
# CHECK:           "jlir.goto"(%[[VAL_12]], %[[VAL_13]], %[[VAL_14]])[^bb4] : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK:         ^bb3:
# CHECK:           %[[VAL_15:.*]] = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK:           %[[VAL_16:.*]] = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK:           %[[VAL_17:.*]] = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK:           "jlir.goto"(%[[VAL_15]], %[[VAL_16]], %[[VAL_17]])[^bb4] : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK:         ^bb4(%[[VAL_18:.*]]: !jlir.Bool, %[[VAL_19:.*]]: !jlir.Int64, %[[VAL_20:.*]]: !jlir.Int64):
# CHECK:           %[[VAL_21:.*]] = "jlir.constant"() {value = #jlir<"#<intrinsic #43 not_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK:           %[[VAL_22:.*]] = "jlir.call"(%[[VAL_21]], %[[VAL_18]]) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Bool) -> !jlir.Bool
# CHECK:           %[[VAL_23:.*]] = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK:           %[[VAL_24:.*]] = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK:           "jlir.gotoifnot"(%[[VAL_22]], %[[VAL_24]], %[[VAL_19]], %[[VAL_20]], %[[VAL_23]])[^bb10, ^bb5] {operand_segment_sizes = dense<[1, 1, 3]> : vector<3xi32>} : (!jlir.Bool, !jlir.Int64, !jlir.Int64, !jlir.Int64, !jlir.Int64) -> ()
# CHECK:         ^bb5(%[[VAL_25:.*]]: !jlir.Int64, %[[VAL_26:.*]]: !jlir.Int64, %[[VAL_27:.*]]: !jlir.Int64):
# CHECK:           %[[VAL_28:.*]] = "jlir.constant"() {value = #jlir<"#<intrinsic #2 add_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK:           %[[VAL_29:.*]] = "jlir.call"(%[[VAL_28]], %[[VAL_27]], %[[VAL_25]]) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK:           %[[VAL_30:.*]] = "jlir.constant"() {value = #jlir<"===">} : () -> !jlir<"typeof(===)">
# CHECK:           %[[VAL_31:.*]] = "jlir.call"(%[[VAL_30]], %[[VAL_26]], %[[VAL_7]]) : (!jlir<"typeof(===)">, !jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK:           "jlir.gotoifnot"(%[[VAL_31]])[^bb7, ^bb6] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK:         ^bb6:
# CHECK:           %[[VAL_32:.*]] = "jlir.constant"() {value = #jlir.nothing} : () -> !jlir.Nothing
# CHECK:           %[[VAL_33:.*]] = "jlir.undef"() : () -> !jlir.Int64
# CHECK:           %[[VAL_34:.*]] = "jlir.undef"() : () -> !jlir.Int64
# CHECK:           %[[VAL_35:.*]] = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK:           "jlir.goto"(%[[VAL_33]], %[[VAL_34]], %[[VAL_35]])[^bb8] : (!jlir.Int64, !jlir.Int64, !jlir.Bool) -> ()
# CHECK:         ^bb7:
# CHECK:           %[[VAL_36:.*]] = "jlir.constant"() {value = #jlir<"#<intrinsic #2 add_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK:           %[[VAL_37:.*]] = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK:           %[[VAL_38:.*]] = "jlir.call"(%[[VAL_36]], %[[VAL_26]], %[[VAL_37]]) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK:           %[[VAL_39:.*]] = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK:           "jlir.goto"(%[[VAL_38]], %[[VAL_38]], %[[VAL_39]])[^bb8] : (!jlir.Int64, !jlir.Int64, !jlir.Bool) -> ()
# CHECK:         ^bb8(%[[VAL_40:.*]]: !jlir.Int64, %[[VAL_41:.*]]: !jlir.Int64, %[[VAL_42:.*]]: !jlir.Bool):
# CHECK:           %[[VAL_43:.*]] = "jlir.constant"() {value = #jlir<"#<intrinsic #43 not_int>">} : () -> !jlir<"typeof(Core.IntrinsicFunction)">
# CHECK:           %[[VAL_44:.*]] = "jlir.call"(%[[VAL_43]], %[[VAL_42]]) : (!jlir<"typeof(Core.IntrinsicFunction)">, !jlir.Bool) -> !jlir.Bool
# CHECK:           "jlir.gotoifnot"(%[[VAL_44]], %[[VAL_29]])[^bb10, ^bb9] {operand_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>} : (!jlir.Bool, !jlir.Int64) -> ()
# CHECK:         ^bb9:
# CHECK:           "jlir.goto"(%[[VAL_40]], %[[VAL_41]], %[[VAL_29]])[^bb5] : (!jlir.Int64, !jlir.Int64, !jlir.Int64) -> ()
# CHECK:         ^bb10(%[[VAL_45:.*]]: !jlir.Int64):
# CHECK:           "jlir.return"(%[[VAL_45]]) : (!jlir.Int64) -> ()
# CHECK:         }
