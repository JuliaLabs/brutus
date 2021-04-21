module Compiler

using MLIR
import MLIR.IR as JLIR

include("opbuilder.jl")
include("jlirgen.jl")
include("codegen.jl")

end # module
