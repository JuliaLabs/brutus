module Compiler

using MLIR
import MLIR.IR as JLIR
import Base: push!

include("opbuilder.jl")
include("jlirgen.jl")
include("codegen.jl")

end # module
