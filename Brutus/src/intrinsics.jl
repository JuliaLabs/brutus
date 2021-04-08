abstract type BrutusIntrinsic <: Function end

struct Matmul! <: BrutusIntrinsic end
@noinline (::Matmul!)(A, B, C) = Base.inferencebarrier(nothing)::Nothing
const matmul! = Matmul!()