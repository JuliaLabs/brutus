# RUN: julia --startup-file=no %s 2>&1 | FileCheck %s

import Brutus

emit_translated(f, tt...) =
    Brutus.emit(typeof(f), tt,
                emit_llvm=false,
                dump_options=[Brutus.DumpTranslated])

f(x) = x
emit_translated(f, Int64)
# CHECK: func @f(%arg0: !jlir<"typeof(Main.f)">, %arg1: !jlir.Int64) -> !jlir.Int64
# CHECK:   "jlir.goto"()[^bb1] : () -> ()
# CHECK: ^bb1:
# CHECK:   "jlir.return"(%arg1) : (!jlir.Int64) -> ()

f() = nothing
emit_translated(f)
# CHECK: func @f(%arg0: !jlir<"typeof(Main.f)">) -> !jlir.Nothing
# CHECK:   "jlir.goto"()[^bb1] : () -> ()
# CHECK: ^bb1:
# CHECK:   %0 = "jlir.constant"() {value = #jlir.nothing} : () -> !jlir.Nothing
# CHECK:   "jlir.return"(%0) : (!jlir.Nothing) -> ()

f() = return
emit_translated(f)
# CHECK: func @f(%arg0: !jlir<"typeof(Main.f)">) -> !jlir.Nothing
# CHECK:   "jlir.goto"()[^bb1] : () -> ()
# CHECK: ^bb1:
# CHECK:   %0 = "jlir.constant"() {value = #jlir.nothing} : () -> !jlir.Nothing
# CHECK:   "jlir.return"(%0) : (!jlir.Nothing) -> ()

f() = return 2
emit_translated(f)
# CHECK: func @f(%arg0: !jlir<"typeof(Main.f)">) -> !jlir.Int64 {
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
# 4 │   %4 = Base.slt_int(%3, 0)::Bool
#   └──      goto #4 if not %4
# 5 3 ─      goto #2
# 7 4 ─      return %3
###
emit_translated(labels, Int64)
# CHECK: func @labels(%arg0: !jlir<"typeof(Main.labels)">, %arg1: !jlir.Int64) -> !jlir.Int64
# CHECK:   "jlir.goto"()[^bb1] : () -> ()
# CHECK: ^bb1:
# CHECK:   "jlir.goto"()[^bb2(%arg1 : !jlir.Int64)] : () -> ()
# CHECK: ^bb2(%0: !jlir.Int64):
# CHECK:   %1 = "jlir.constant"() {value = #jlir<"#<intrinsic #2 add_int>">} : () -> !jlir.Core.IntrinsicFunction
# CHECK:   %2 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK:   %3 = "jlir.call"(%1, %0, %2) : (!jlir.Core.IntrinsicFunction, !jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK:   %4 = "jlir.constant"() {value = #jlir<"#<intrinsic #27 slt_int>">} : () -> !jlir.Core.IntrinsicFunction
# CHECK:   %5 = "jlir.constant"() {value = #jlir<"0">} : () -> !jlir.Int64
# CHECK:   %6 = "jlir.call"(%4, %3, %5) : (!jlir.Core.IntrinsicFunction, !jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK:   "jlir.gotoifnot"(%6)[^bb4, ^bb3] : (!jlir.Bool) -> ()
# CHECK: ^bb3:
# CHECK:   "jlir.goto"()[^bb2(%3 : !jlir.Int64)] : () -> ()
# CHECK: ^bb4:
# CHECK:   "jlir.return"(%3) : (!jlir.Int64) -> ()

function branches(c)
    if c
        return c
    else
        return !c
    end
end
emit_translated(branches, Bool)
# CHECK:  func @branches(%arg0: !jlir<"typeof(Main.branches)">, %arg1: !jlir.Bool) -> !jlir.Bool
# CHECK:    "jlir.goto"()[^bb1] : () -> ()
# CHECK:  ^bb1:
# CHECK:    "jlir.gotoifnot"(%arg1)[^bb3, ^bb2] : (!jlir.Bool) -> ()
# CHECK:  ^bb2:
# CHECK:    "jlir.return"(%arg1) : (!jlir.Bool) -> ()
# CHECK:  ^bb3:
# CHECK:    %0 = "jlir.constant"() {value = #jlir<"#<intrinsic #44 not_int>">} : () -> !jlir.Core.IntrinsicFunction
# CHECK:    %1 = "jlir.call"(%0, %arg1) : (!jlir.Core.IntrinsicFunction, !jlir.Bool) -> !jlir.Bool
# CHECK:    "jlir.return"(%1) : (!jlir.Bool) -> ()

function loop(N)
    acc = 1
    for i in 1:N
        acc += i
    end
    return acc
end
emit_translated(loop, Int64)
# CHECK:  func @loop(%arg0: !jlir<"typeof(Main.loop)">, %arg1: !jlir.Int64) -> !jlir.Int64
# CHECK:    "jlir.goto"()[^bb1] : () -> ()
# CHECK:  ^bb1:
# CHECK:    %0 = "jlir.constant"() {value = #jlir<"#<intrinsic #29 sle_int>">} : () -> !jlir.Core.IntrinsicFunction
# CHECK:    %1 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK:    %2 = "jlir.call"(%0, %1, %arg1) : (!jlir.Core.IntrinsicFunction, !jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK:    %3 = "jlir.constant"() {value = #jlir<"typeof(ifelse)()">} : () -> !jlir<"typeof(ifelse)">
# CHECK:    %4 = "jlir.constant"() {value = #jlir<"0">} : () -> !jlir.Int64
# CHECK:    %5 = "jlir.call"(%3, %2, %arg1, %4) : (!jlir<"typeof(ifelse)">, !jlir.Bool, !jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK:    %6 = "jlir.constant"() {value = #jlir<"#<intrinsic #27 slt_int>">} : () -> !jlir.Core.IntrinsicFunction
# CHECK:    %7 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK:    %8 = "jlir.call"(%6, %5, %7) : (!jlir.Core.IntrinsicFunction, !jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK:    "jlir.gotoifnot"(%8)[^bb3, ^bb2] : (!jlir.Bool) -> ()
# CHECK:  ^bb2:
# CHECK:    %9 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK:    %10 = "jlir.undef"() : () -> !jlir.Int64
# CHECK:    %11 = "jlir.undef"() : () -> !jlir.Int64
# CHECK:    "jlir.goto"()[^bb4(%9, %10, %11 : !jlir.Bool, !jlir.Int64, !jlir.Int64)] : () -> ()
# CHECK:  ^bb3:
# CHECK:    %12 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK:    %13 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK:    %14 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK:    "jlir.goto"()[^bb4(%12, %13, %14 : !jlir.Bool, !jlir.Int64, !jlir.Int64)] : () -> ()
# CHECK:  ^bb4(%15: !jlir.Bool, %16: !jlir.Int64, %17: !jlir.Int64):
# CHECK:    %18 = "jlir.constant"() {value = #jlir<"#<intrinsic #44 not_int>">} : () -> !jlir.Core.IntrinsicFunction
# CHECK:    %19 = "jlir.call"(%18, %15) : (!jlir.Core.IntrinsicFunction, !jlir.Bool) -> !jlir.Bool
# CHECK:    %20 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK:    %21 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK:    "jlir.gotoifnot"(%19)[^bb10(%21 : !jlir.Int64), ^bb5(%20, %16, %17 : !jlir.Int64, !jlir.Int64, !jlir.Int64)] : (!jlir.Bool) -> ()
# CHECK:  ^bb5(%22: !jlir.Int64, %23: !jlir.Int64, %24: !jlir.Int64):
# CHECK:    %25 = "jlir.constant"() {value = #jlir<"#<intrinsic #2 add_int>">} : () -> !jlir.Core.IntrinsicFunction
# CHECK:    %26 = "jlir.call"(%25, %22, %23) : (!jlir.Core.IntrinsicFunction, !jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK:    %27 = "jlir.constant"() {value = #jlir<"typeof(===)()">} : () -> !jlir<"typeof(===)">
# CHECK:    %28 = "jlir.call"(%27, %24, %5) : (!jlir<"typeof(===)">, !jlir.Int64, !jlir.Int64) -> !jlir.Bool
# CHECK:    "jlir.gotoifnot"(%28)[^bb7, ^bb6] : (!jlir.Bool) -> ()
# CHECK:  ^bb6:
# CHECK:    %29 = "jlir.undef"() : () -> !jlir.Int64
# CHECK:    %30 = "jlir.undef"() : () -> !jlir.Int64
# CHECK:    %31 = "jlir.constant"() {value = #jlir.true} : () -> !jlir.Bool
# CHECK:    "jlir.goto"()[^bb8(%29, %30, %31 : !jlir.Int64, !jlir.Int64, !jlir.Bool)] : () -> ()
# CHECK:  ^bb7:
# CHECK:    %32 = "jlir.constant"() {value = #jlir<"#<intrinsic #2 add_int>">} : () -> !jlir.Core.IntrinsicFunction
# CHECK:    %33 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK:    %34 = "jlir.call"(%32, %24, %33) : (!jlir.Core.IntrinsicFunction, !jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK:    %35 = "jlir.constant"() {value = #jlir.false} : () -> !jlir.Bool
# CHECK:    "jlir.goto"()[^bb8(%34, %34, %35 : !jlir.Int64, !jlir.Int64, !jlir.Bool)] : () -> ()
# CHECK:  ^bb8(%36: !jlir.Int64, %37: !jlir.Int64, %38: !jlir.Bool):
# CHECK:    %39 = "jlir.constant"() {value = #jlir<"#<intrinsic #44 not_int>">} : () -> !jlir.Core.IntrinsicFunction
# CHECK:    %40 = "jlir.call"(%39, %38) : (!jlir.Core.IntrinsicFunction, !jlir.Bool) -> !jlir.Bool
# CHECK:    "jlir.gotoifnot"(%40)[^bb10(%26 : !jlir.Int64), ^bb9] : (!jlir.Bool) -> ()
# CHECK:  ^bb9:
# CHECK:    "jlir.goto"()[^bb5(%26, %36, %37 : !jlir.Int64, !jlir.Int64, !jlir.Int64)] : () -> ()
# CHECK:  ^bb10(%41: !jlir.Int64):
# CHECK:    "jlir.return"(%41) : (!jlir.Int64) -> ()

function calls()
    f = rand(Bool) ? (+) : (-)
    return f(1, 1)
end
emit_translated(calls)
# CHECK:  func @calls(%arg0: !jlir<"typeof(Main.calls)">) -> !jlir.Any
# CHECK:    "jlir.goto"()[^bb1] : () -> ()
# CHECK:  ^bb1:
# CHECK:    %0 = "jlir.constant"() {value = #jlir.Bool} : () -> !jlir.DataType
# CHECK:    %1 = "jlir.invoke"(%0) {methodInstance = #jlir<"rand(Type{Bool})">} : (!jlir.DataType) -> !jlir.Any
# CHECK:    "jlir.gotoifnot"(%1)[^bb3, ^bb2] : (!jlir.Any) -> ()
# CHECK:  ^bb2:
# CHECK:    %2 = "jlir.constant"() {value = #jlir<"typeof(Base.:(+))()">} : () -> !jlir<"typeof(Base.:(+))">
# CHECK:    %3 = "jlir.pi"(%2) : (!jlir<"typeof(Base.:(+))">) -> !jlir<"Union{typeof(Base.:(+)), typeof(Base.:(-))}">
# CHECK:    "jlir.goto"()[^bb4(%3 : !jlir<"Union{typeof(Base.:(+)), typeof(Base.:(-))}">)] : () -> ()
# CHECK:  ^bb3:
# CHECK:    %4 = "jlir.constant"() {value = #jlir<"typeof(Base.:(-))()">} : () -> !jlir<"typeof(Base.:(-))">
# CHECK:    %5 = "jlir.pi"(%4) : (!jlir<"typeof(Base.:(-))">) -> !jlir<"Union{typeof(Base.:(+)), typeof(Base.:(-))}">
# CHECK:    "jlir.goto"()[^bb4(%5 : !jlir<"Union{typeof(Base.:(+)), typeof(Base.:(-))}">)] : () -> ()
# CHECK:  ^bb4(%6: !jlir<"Union{typeof(Base.:(+)), typeof(Base.:(-))}">):
# CHECK:    %7 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK:    %8 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK:    %9 = "jlir.call"(%6, %7, %8) : (!jlir<"Union{typeof(Base.:(+)), typeof(Base.:(-))}">, !jlir.Int64, !jlir.Int64) -> !jlir.Any
# CHECK:    "jlir.return"(%9) : (!jlir.Any) -> ()

struct A
    x
end
(a::A)(y) = a.x + y
a = A(10)
emit_translated(a, Int64)
# CHECK:  func @A(%arg0: !jlir.Main.A, %arg1: !jlir.Int64) -> !jlir.Any
# CHECK:    "jlir.goto"()[^bb1] : () -> ()
# CHECK:  ^bb1:
# CHECK:    %0 = "jlir.constant"() {value = #jlir<"typeof(getfield)()">} : () -> !jlir<"typeof(getfield)">
# CHECK:    %1 = "jlir.constant"() {value = #jlir<":x">} : () -> !jlir.Symbol
# CHECK:    %2 = "jlir.call"(%0, %arg0, %1) : (!jlir<"typeof(getfield)">, !jlir.Main.A, !jlir.Symbol) -> !jlir.Any
# CHECK:    %3 = "jlir.constant"() {value = #jlir<"typeof(Base.:(+))()">} : () -> !jlir<"typeof(Base.:(+))">
# CHECK:    %4 = "jlir.call"(%3, %2, %arg1) : (!jlir<"typeof(Base.:(+))">, !jlir.Any, !jlir.Int64) -> !jlir.Any
# CHECK:    "jlir.return"(%4) : (!jlir.Any) -> ()

function haspi(x::Union{Int64, Float64})
    if x isa Int64
        return x + 1
    end
end
emit_translated(haspi, Union{Int64, Float64})
# CHECK:  func @haspi(%arg0: !jlir<"typeof(Main.haspi)">, %arg1: !jlir<"Union{Float64, Int64}">) -> !jlir<"Union{Nothing, Int64}">
# CHECK:    "jlir.goto"()[^bb1] : () -> ()
# CHECK:  ^bb1:
# CHECK:    %0 = "jlir.constant"() {value = #jlir<"typeof(isa)()">} : () -> !jlir<"typeof(isa)">
# CHECK:    %1 = "jlir.constant"() {value = #jlir.Int64} : () -> !jlir.DataType
# CHECK:    %2 = "jlir.call"(%0, %arg1, %1) : (!jlir<"typeof(isa)">, !jlir<"Union{Float64, Int64}">
# CHECK:    "jlir.gotoifnot"(%2)[^bb3, ^bb2] : (!jlir.Bool) -> ()
# CHECK:  ^bb2:
# CHECK:    %3 = "jlir.pi"(%arg1) : (!jlir<"Union{Float64, Int64}">) -> !jlir.Int64
# CHECK:    %4 = "jlir.constant"() {value = #jlir<"#<intrinsic #2 add_int>">} : () -> !jlir.Core.IntrinsicFunction
# CHECK:    %5 = "jlir.constant"() {value = #jlir<"1">} : () -> !jlir.Int64
# CHECK:    %6 = "jlir.call"(%4, %3, %5) : (!jlir.Core.IntrinsicFunction, !jlir.Int64, !jlir.Int64) -> !jlir.Int64
# CHECK:    %7 = "jlir.pi"(%6) : (!jlir.Int64) -> !jlir<"Union{Nothing, Int64}">
# CHECK:    "jlir.return"(%7) : (!jlir<"Union{Nothing, Int64}">) -> ()
# CHECK:  ^bb3:
# CHECK:    %8 = "jlir.constant"() {value = #jlir.nothing} : () -> !jlir.Nothing
# CHECK:    %9 = "jlir.pi"(%8) : (!jlir.Nothing) -> !jlir<"Union{Nothing, Int64}">
# CHECK:    "jlir.return"(%9) : (!jlir<"Union{Nothing, Int64}">) -> ()

# has the terminator unreachable
hasunreachable(x::Float64) = sqrt(x)
emit_translated(hasunreachable, Float64)
# CHECK:  func @hasunreachable(%arg0: !jlir<"typeof(Main.hasunreachable)">, %arg1: !jlir.Float64) -> !jlir.Float64
# CHECK:    "jlir.goto"()[^bb1] : () -> ()
# CHECK:  ^bb1:
# CHECK:    %0 = "jlir.constant"() {value = #jlir<"#<intrinsic #33 lt_float>">} : () -> !jlir.Core.IntrinsicFunction
# CHECK:    %1 = "jlir.constant"() {value = #jlir<"0">} : () -> !jlir.Float64
# CHECK:    %2 = "jlir.call"(%0, %arg1, %1) : (!jlir.Core.IntrinsicFunction, !jlir.Float64, !jlir.Float64) -> !jlir.Bool
# CHECK:    "jlir.gotoifnot"(%2)[^bb3, ^bb2] : (!jlir.Bool) -> ()
# CHECK:  ^bb2:	// pred: ^bb1
# CHECK:    %3 = "jlir.constant"() {value = #jlir<":sqrt">} : () -> !jlir.Symbol
# CHECK:    %4 = "jlir.invoke"(%3, %arg1) {methodInstance = #jlir<"throw_complex_domainerror(Symbol, Float64)">} : (!jlir.Symbol, !jlir.Float64) -> !jlir.Any
# CHECK:    %5 = "jlir.undef"() : () -> !jlir.Float64
# CHECK:    "jlir.return"(%5) : (!jlir.Float64) -> ()
# CHECK:  ^bb3:	// pred: ^bb1
# CHECK:    %6 = "jlir.constant"() {value = #jlir<"#<intrinsic #78 sqrt_llvm>">} : () -> !jlir.Core.IntrinsicFunction
# CHECK:    %7 = "jlir.call"(%6, %arg1) : (!jlir.Core.IntrinsicFunction, !jlir.Float64) -> !jlir.Float64
# CHECK:    "jlir.goto"()[^bb4] : () -> ()
# CHECK:  ^bb4:	// pred: ^bb3
# CHECK:    "jlir.return"(%7) : (!jlir.Float64) -> ()
