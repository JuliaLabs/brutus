# RUN: julia --startup-file=no %s 2>&1 | FileCheck %s

import Brutus

emit_optimized(f, tt...) =
    Brutus.emit(typeof(f), tt,
                emit_fptr=false,
                dump_options=[Brutus.DumpOptimized])

f(x) = x
emit_optimized(f, Int64)
# CHECK: func @f(%arg0: !jlir<"typeof(Main.f)">, %arg1: !jlir.Int64) -> !jlir.Int64
# CHECK:   "jlir.goto"()[^bb1] : () -> ()
# CHECK: ^bb1:
# CHECK:   "jlir.return"(%arg1) : (!jlir.Int64) -> ()

f() = nothing
emit_optimized(f)
# CHECK: func @f(%arg0: !jlir<"typeof(Main.f)">) -> !jlir.Nothing
# CHECK:   "jlir.goto"()[^bb1] : () -> ()
# CHECK: ^bb1:
# CHECK:   %0 = "jlir.constant"() {value = #jlir.nothing} : () -> !jlir.Nothing
# CHECK:   "jlir.return"(%0) : (!jlir.Nothing) -> ()

f() = return
emit_optimized(f)
# CHECK: func @f(%arg0: !jlir<"typeof(Main.f)">) -> !jlir.Nothing
# CHECK:   "jlir.goto"()[^bb1] : () -> ()
# CHECK: ^bb1:
# CHECK:   %0 = "jlir.constant"() {value = #jlir.nothing} : () -> !jlir.Nothing
# CHECK:   "jlir.return"(%0) : (!jlir.Nothing) -> ()

f() = return 2
emit_optimized(f)
# CHECK: func @f(%arg0: !jlir<"typeof(Main.f)">) -> !jlir.Int64
# CHECK:   "jlir.goto"()[^bb1] : () -> ()
# CHECK: ^bb1:
# CHECK:   %0 = "jlir.constant"() {value = #jlir<"2">} : () -> !jlir.Int64
# CHECK:   "jlir.return"(%0) : (!jlir.Int64) -> ()

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
# 4 │   %4 = Base.jlir.slt_int(%3, 0)::Bool
#   └──      goto #4 if not %4
# 5 3 ─      goto #2
# 7 4 ─      return %3
###
emit_optimized(labels, Int64)
# CHECK: func @labels(%arg0: !jlir<"typeof(Main.labels)">, %arg1: !jlir.Int64) -> !jlir.Int64
# CHECK:   "jlir.goto"()[^bb1] : () -> ()
# CHECK: ^bb1:
# CHECK:   "jlir.goto"(%arg1)[^bb2] : (!jlir.Int64) -> ()
# CHECK: ^bb2(%0: !jlir.Int64):
# CHECK:   %1 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK:   %2 = "jlir.add_int"(%0, %1) : (!jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK:   %3 = "jlir.constant"() {value = #jlir<"0">} : () -> !jlir.Int64
# CHECK:   %4 = "jlir.slt_int"(%2, %3) : (!jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK:   "jlir.gotoifnot"(%4)[^bb4, ^bb3] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK: ^bb3:
# CHECK:   "jlir.goto"(%2)[^bb2] : (!jlir.Int64) -> ()
# CHECK: ^bb4:
# CHECK:   "jlir.return"(%2) : (!jlir.Int64) -> ()

function branches(c)
    if c
        return c
    else
        return !c
    end
end
emit_optimized(branches, Bool)
# CHECK: func @branches(%arg0: !jlir<"typeof(Main.branches)">, %arg1: !jlir.Bool) -> !jlir.Bool
# CHECK:   "jlir.goto"()[^bb1] : () -> ()
# CHECK: ^bb1:
# CHECK:   "jlir.gotoifnot"(%arg1)[^bb3, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK: ^bb2:
# CHECK:   "jlir.return"(%arg1) : (!jlir.Bool) -> ()
# CHECK: ^bb3:
# CHECK:   %0 = "jlir.not_int"(%arg1) : (!jlir.Bool) -> !jlir.Bool
# CHECK:   "jlir.return"(%0) : (!jlir.Bool) -> ()

function loop(N)
    acc = 0
    for i in 1:N
        acc += i
    end
    return acc
end
emit_optimized(loop, Int64)
# CHECK: func @loop(%arg0: !jlir<"typeof(Main.loop)">, %arg1: !jlir.Int64) -> !jlir.Int64
# CHECK:   "jlir.goto"()[^bb1] : () -> ()
# CHECK: ^bb1:
# CHECK:   %0 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK:   %1 = "jlir.sle_int"(%0, %arg1) : (!jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK:   %2 = "jlir.constant"() {value = #jlir<"0">} : () -> !jlir.Int64
# CHECK:   %3 = "jlir.ifelse"(%1, %arg1, %2) : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK:   %4 = "jlir.slt_int"(%3, %0) : (!jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK:   "jlir.gotoifnot"(%4)[^bb3, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK: ^bb2:
# CHECK:   %5 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK:   %6 = "jlir.undef"() : () -> !jlir.Int64
# CHECK:   %7 = "jlir.undef"() : () -> !jlir.Int64
# CHECK:   "jlir.goto"(%5, %6, %7)[^bb4] : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK: ^bb3:
# CHECK:   %8 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK:   "jlir.goto"(%8, %0, %0)[^bb4] : (!jlir.Bool, !jlir.Int64, !jlir.Int64) -> ()
# CHECK: ^bb4(%9: !jlir.Bool, %10: !jlir.Int64, %11: !jlir.Int64):
# CHECK:   %12 = "jlir.not_int"(%9) : (!jlir.Bool) -> !jlir.Bool
# CHECK:   "jlir.gotoifnot"(%12, %2, %2, %10, %11)[^bb10, ^bb5] {operand_segment_sizes = dense<[1, 1, 3]> : vector<3xi32>} : (!jlir.Bool, !jlir.Int64, !jlir.Int64, !jlir.Int64, !jlir.Int64) -> ()
# CHECK: ^bb5(%13: !jlir.Int64, %14: !jlir.Int64, %15: !jlir.Int64):
# CHECK:   %16 = "jlir.add_int"(%13, %14) : (!jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK:   %17 = "jlir.==="(%15, %3) : (!jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK:   "jlir.gotoifnot"(%17)[^bb7, ^bb6] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK: ^bb6:
# CHECK:   %18 = "jlir.undef"() : () -> !jlir.Int64
# CHECK:   %19 = "jlir.undef"() : () -> !jlir.Int64
# CHECK:   %20 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK:   "jlir.goto"(%18, %19, %20)[^bb8] : (!jlir.Int64, !jlir.Int64, !jlir.Bool) -> ()
# CHECK: ^bb7:
# CHECK:   %21 = "jlir.add_int"(%15, %0) : (!jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK:   %22 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK:   "jlir.goto"(%21, %21, %22)[^bb8] : (!jlir.Int64, !jlir.Int64, !jlir.Bool) -> ()
# CHECK: ^bb8(%23: !jlir.Int64, %24: !jlir.Int64, %25: !jlir.Bool):
# CHECK:   %26 = "jlir.not_int"(%25) : (!jlir.Bool) -> !jlir.Bool
# CHECK:   "jlir.gotoifnot"(%26, %16)[^bb10, ^bb9] {operand_segment_sizes = dense<[1, 1, 0]> : vector<3xi32>} : (!jlir.Bool, !jlir.Int64) -> ()
# CHECK: ^bb9:
# CHECK:   "jlir.goto"(%16, %23, %24)[^bb5] : (!jlir.Int64, !jlir.Int64, !jlir.Int64) -> ()
# CHECK: ^bb10(%27: !jlir.Int64):
# CHECK:   "jlir.return"(%27) : (!jlir.Int64) -> ()

function calls()
    f = rand(Bool) ? (+) : (-)
    return f(1, 1)
end
emit_optimized(calls)
# CHECK: func @calls(%arg0: !jlir<"typeof(Main.calls)">) -> !jlir.Any
# CHECK:   "jlir.goto"()[^bb1] : () -> ()
# CHECK: ^bb1:
# CHECK:   %0 = "jlir.constant"() {value = #jlir.Bool} : () -> !jlir.DataType
# CHECK:   %1 = "jlir.invoke"(%0) {methodInstance = #jlir<"rand(Type{Bool})">} : (!jlir.DataType) -> !jlir.Any
# CHECK:   "jlir.gotoifnot"(%1)[^bb3, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Any) -> ()
# CHECK: ^bb2:
# CHECK:   %2 = "jlir.constant"() {value = #jlir<"typeof(Base.:(+))()">} : () -> !jlir<"typeof(Base.:(+))">
# CHECK:   %3 = "jlir.pi"(%2) : (!jlir<"typeof(Base.:(+))">) -> !jlir<"Union{typeof(Base.:(+)), typeof(Base.:(-))}">
# CHECK:   "jlir.goto"(%3)[^bb4] : (!jlir<"Union{typeof(Base.:(+)), typeof(Base.:(-))}">) -> ()
# CHECK: ^bb3:
# CHECK:   %4 = "jlir.constant"() {value = #jlir<"typeof(Base.:(-))()">} : () -> !jlir<"typeof(Base.:(-))">
# CHECK:   %5 = "jlir.pi"(%4) : (!jlir<"typeof(Base.:(-))">) -> !jlir<"Union{typeof(Base.:(+)), typeof(Base.:(-))}">
# CHECK:   "jlir.goto"(%5)[^bb4] : (!jlir<"Union{typeof(Base.:(+)), typeof(Base.:(-))}">) -> ()
# CHECK: ^bb4(%6: !jlir<"Union{typeof(Base.:(+)), typeof(Base.:(-))}">):
# CHECK:   %7 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK:   %8 = "jlir.call"(%6, %7, %7) : (!jlir<"Union{typeof(Base.:(+)), typeof(Base.:(-))}">, !jlir.Int64, !jlir.Int64) -> !jlir.Any
# CHECK:   "jlir.return"(%8) : (!jlir.Any) -> ()

struct A
    x
end
(a::A)(y) = a.x + y
a = A(10)
emit_optimized(a, Int64)
# CHECK: func @A(%arg0: !jlir.Main.A, %arg1: !jlir.Int64) -> !jlir.Any
# CHECK:   "jlir.goto"()[^bb1] : () -> ()
# CHECK: ^bb1:
# CHECK:   %0 = "jlir.constant"() {value = #jlir<":x">} : () -> !jlir.Symbol
# CHECK:   %1 = "jlir.getfield"(%arg0, %0) : (!jlir.Main.A, !jlir.Symbol) -> !jlir.Any
# CHECK:   %2 = "jlir.constant"() {value = #jlir<"typeof(Base.:(+))()">} : () -> !jlir<"typeof(Base.:(+))">
# CHECK:   %3 = "jlir.call"(%2, %1, %arg1) : (!jlir<"typeof(Base.:(+))">, !jlir.Any, !jlir.Int64) -> !jlir.Any
# CHECK:   "jlir.return"(%3) : (!jlir.Any) -> ()

function haspi(x::Union{Int64, Float64})
    if x isa Int64
        return x + 1
    end
end
emit_optimized(haspi, Union{Int64, Float64})
# CHECK: func @haspi(%arg0: !jlir<"typeof(Main.haspi)">, %arg1: !jlir<"Union{Float64, Int64}">) -> !jlir<"Union{Nothing, Int64}">
# CHECK:   "jlir.goto"()[^bb1] : () -> ()
# CHECK: ^bb1:
# CHECK:   %0 = "jlir.constant"() {value = #jlir.Int64} : () -> !jlir.DataType
# CHECK:   %1 = "jlir.isa"(%arg1, %0) : (!jlir<"Union{Float64, Int64}">, !jlir.DataType) -> !jlir.Bool
# CHECK:   "jlir.gotoifnot"(%1)[^bb3, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (!jlir.Bool) -> ()
# CHECK: ^bb2:
# CHECK:   %2 = "jlir.pi"(%arg1) : (!jlir<"Union{Float64, Int64}">) -> !jlir.Int64
# CHECK:   %3 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK:   %4 = "jlir.add_int"(%2, %3) : (!jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK:   %5 = "jlir.pi"(%4) : (!jlir.Int64) -> !jlir<"Union{Nothing, Int64}">
# CHECK:   "jlir.return"(%5) : (!jlir<"Union{Nothing, Int64}">) -> ()
# CHECK: ^bb3:
# CHECK:   %6 = "jlir.constant"() {value = #jlir.nothing} : () -> !jlir.Nothing
# CHECK:   %7 = "jlir.pi"(%6) : (!jlir.Nothing) -> !jlir<"Union{Nothing, Int64}">
# CHECK:   "jlir.return"(%7) : (!jlir<"Union{Nothing, Int64}">) -> ()
