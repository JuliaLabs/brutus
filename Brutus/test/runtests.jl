using Brutus
using Test

@testset "Brutus" begin

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

function gauss(N)
    acc = 0
    for i in 1:N
        acc += i
    end
    return acc
end
@test Brutus.call(gauss, 0) == gauss(0) == 0
@test Brutus.call(gauss, 1000) == gauss(1000) == 500500

@test Brutus.call(Brutus.call(identity, identity), identity)(identity) === identity

index(A, i) = A[i]
index(A, i, j) = A[i, j]
index(A, i, j, k) = A[i, j, k]

@testset "3D indexing" begin
    thunk = Brutus.emit(index, Array{Int,3}, Int, Int, Int)

    arr = Int[i*j*k for i in 1:3, j in 1:5, k in 1:7]
    for i in 1:3
        for j in 1:5
            for k in 1:7
                @test arr[i,j,k] == thunk(arr, i, j, k)
            end
        end
    end
end

function customsum(A)
    acc = 0
    for i in CartesianIndices(A)
        acc += A[i]
    end
    return acc
end
for array in [rand(Int64, 2, 3), rand(Int64, 2, 3)]
    for i in 1:length(array)
        @test Brutus.call(index, array, i) == index(array, i)
    end
    for i in CartesianIndices(array)
        i = convert(Tuple, i)
        @test Brutus.call(index, array, i...) == index(array, i...)
    end
    @test Brutus.call(customsum, array) == customsum(array)
end
end
# TODO: arrays with floating point elements
