using Brutus
using Test

@test Brutus.call(identity, 1) == identity(1)
@test Brutus.call(identity, 3.141592653) == identity(3.141592653)
# @test Brutus.call(identity, pi) == identity(pi)
@test Brutus.call(identity, true) == identity(true)

@test Brutus.call(+, 1, 2) == 1 + 2
@test Brutus.call(+, 1.1, 2.2) == 1.1 + 2.2
# @test Brutus.call(+, true, false) == true + false

constant1() = 1
@test Brutus.call(constant1) == constant1()
constant2() = 3.141592653
@test Brutus.call(constant2) == constant2()
constant3() = pi
@test Brutus.call(constant3) == constant3()
constant4() = true
@test Brutus.call(constant4) == constant4()
