module JuliaCodegen

using Brutus
using MLIR

function gauss(N)
    acc = 0
    for i in 1:N
        acc += i
    end
    return acc
end

mi = Brutus.get_methodinstance(Tuple{typeof(gauss), Int})
ir_code, ret = Brutus.code_ircode(mi)
ft = Brutus.Compiler.create_func_op(ir_code, ret, "gauss")

end # module
