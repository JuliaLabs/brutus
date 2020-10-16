emit_optimized(f, tt...) =
    emit(typeof(f), tt,
         emit_fptr=false,
         dump_options=[Brutus.DumpCanonicalized])

emit_lowered(f, tt...) =
    emit(typeof(f), tt,
         emit_fptr=false, # TODO: change to true when ready
         dump_options=[Brutus.DumpLoweredToStd,
                       Brutus.DumpLoweredToLLVM])

emit_translated(f, tt...) =
    emit(typeof(f), tt,
         emit_fptr=false,
         dump_options=[Brutus.DumpTranslated])

emit_llvm(f, tt...) =
    emit(typeof(f), tt,
         emit_fptr=false, # TODO: change to true when ready
         dump_options=[Brutus.DumpLoweredToLLVM,
                       Brutus.DumpTranslateToLLVM]

function lit(emit_sym)
    @assert length(ARGS) == 1
    temp = Module(:Main)
    expr = quote
            include(fname) = Core.include(@__MODULE__, fname)
            import Brutus
            emit(args...;kwargs...) = Brutus.$(emit_sym)(args...;kwargs...)
            include($(ARGS[1]))
    end
    Base.eval(temp, Expr(:toplevel, expr))
end
