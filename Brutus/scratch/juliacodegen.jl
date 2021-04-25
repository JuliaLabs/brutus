module JuliaCodegen

using Brutus
using MLIR

println("\n---- brutus_id ----\n")
function brutus_id(N)
    return N
end

v = Brutus.call(brutus_id, 5)
display(v)

println("\n---- brutus_add ----\n")

function brutus_add(N1, N2)
    return N1 + N2
end

v = Brutus.call(brutus_add, 5.0, 10.0)
display(v)

#println("\n---- switch ----\n")
#
#function switch(N)
#    N > 10 ? 5 : 10
#end
#
#mi = Brutus.get_methodinstance(Tuple{typeof(switch), Int})
#ir_code, rt = Brutus.code_ircode(mi)
#display(ir_code)
#jlir = Brutus.Compiler.codegen_jlir(ir_code, rt, "switch")
#display(jlir)
#Brutus.Compiler.canonicalize!(jlir)
#display(jlir)
#
#println("\n---- gauss ----\n")
#
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
#display(ir_code)
#mod = Brutus.Compiler.codegen_jlir(ir_code, rt, "gauss")
#MLIR.IR.dump(mod)

end # module
