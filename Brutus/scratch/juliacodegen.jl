module JuliaCodegen

using Brutus
using MLIR

function gauss(N)
    k = 0
    for i in 1 : N
        k += i
    end
    return k
end

mi = Brutus.get_methodinstance(Tuple{typeof(gauss), Int})
ir_code, ret = Brutus.code_ircode(mi)
mod = Brutus.Compiler.emit_jlir(ir_code, ret, "gauss")

end # module
