module JuliaCodegen

using Brutus
using MLIR

brutus_id(N) = N

mi = Brutus.get_methodinstance(Tuple{typeof(brutus_id), Int})
ir_code, rt = Brutus.code_ircode(mi)
mod = Brutus.Compiler.emit_jlir(ir_code, rt, "brutus_id")

#function gauss(N)
#    k = 0
#    for i in 1 : N
#        k += i
#    end
#    return k
#end
#
#mi = Brutus.get_methodinstance(Tuple{typeof(gauss), Int})
#ir_code, rt = Brutus.code_ircode(mi)
#mod = Brutus.Compiler.emit_jlir(ir_code, rt, "gauss")

end # module
