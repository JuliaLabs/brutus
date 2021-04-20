module Compiler

using MLIR
import MLIR.IR as JLIR

include("jlirgen.jl")
include("opbuilder.jl")
include("codegen.jl")

end # module
