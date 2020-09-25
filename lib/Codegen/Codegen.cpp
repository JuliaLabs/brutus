#include "brutus/brutus.h"
#include "brutus/brutus_internal.h"
#include "brutus/Dialect/Julia/JuliaOps.h"
#include "brutus/Conversion/JLIRToLLVM/JLIRToLLVM.h"
#include "brutus/Conversion/JLIRToStandard/JLIRToStandard.h"

#include "julia.h"

#include "mlir/IR/Verifier.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::jlir;

class jl_mlirctx_t {
public:
    OpBuilder builder;
    MLIRContext *context;
    std::vector<Value> values;
    std::vector<Value> arguments;

    jl_mlirctx_t(MLIRContext *context)
      : builder(context), context(context) { }
};

std::string mi_name(jl_method_instance_t *mi) {
    ios_t str_;
    ios_mem(&str_, 10000);
    JL_STREAM* str = (JL_STREAM*)&str_;

    // jl_printf(str, "%s.", jl_symbol_name(mi->def.method->module->name));
    // // doesn't work well with call overloading
    // jl_static_show_func_sig(str, mi->specTypes);
    jl_static_show(str, mi->specTypes);
    str_.buf[str_.size] = '\0';

    std::string s(str_.buf);
    ios_close(&str_);

    return s;
}

mlir::Value maybe_widen_type(jl_mlirctx_t &ctx, mlir::Location loc,
                        mlir::Value value, jl_datatype_t *expected_type) {
    // widen the type of the value with a PiOp if its type is a subtype of the
    // expected type
    jl_value_t *value_type =
        (jl_value_t*)value.getType().cast<JuliaType>().getDatatype();
    if (!jl_egal(value_type, (jl_value_t*)expected_type)
        && jl_subtype(value_type, (jl_value_t*)expected_type)) {
        auto op = ctx.builder.create<PiOp>(loc, value, expected_type);
        return op.getResult();
    }

    // value was already of expected type, or its type is not a subtype of what
    // was expected and cannot be widened (so detect this mismatch later)
    return value;
}

mlir::Value emit_value(jl_mlirctx_t &ctx, mlir::Location loc,
                       jl_value_t *value, jl_datatype_t *type = nullptr) {
    // check if we have a const globalref
    if (jl_is_globalref(value)) {
        jl_sym_t *s = jl_globalref_name(value);
        jl_binding_t *b = jl_get_binding(jl_globalref_mod(value), s);
        if (b && b->constp) {
            // if (b->deprecated)
            // FIXME: Deprecation warning
            value = b->value;
        }
    }

    // Not to be confused with a raw symbol...
    // that would mean a global variable
    if (jl_is_quotenode(value)) {
        value = jl_fieldref_noalloc(value, 0);
    }

    if (type == nullptr)
        type = (jl_datatype_t*)jl_typeof(value);

    if (jl_is_ssavalue(value)) {
        ssize_t idx = ((jl_ssavalue_t*)value)->id - 1;
        assert(idx >= 0);
        return ctx.values[idx];

    } else if (jl_isa(value, argument_type)) {
        // FIXME: this is unsafe and stupid, but they do have the same layout
        // first argument is function itself, can be needed with call overloading
        // and closures
        ssize_t idx = ((jl_ssavalue_t*)value)->id - 1;
        assert(idx >= 0);
        return (mlir::Value) ctx.arguments[idx];
    } else if (jl_is_globalref(value)) {
        // FIXME: Non-const globalref
        auto op = ctx.builder.create<UnimplementedOp>(loc, type);
        return op.getResult();
    } else {
        auto op = ctx.builder.create<ConstantOp>(loc, value, type);
        return op.getResult();
    }
}

mlir::Value emit_expr(jl_mlirctx_t &ctx, Location &loc, jl_expr_t *expr, jl_datatype_t *type) {
    jl_sym_t *head = expr->head;
    size_t nargs = jl_array_dim0(expr->args);
    jl_value_t **args = (jl_value_t**)jl_array_data(expr->args);

    // from codegen.cpp:
    // if (head == isdefined_sym) {
    // } else if (head == throw_undef_if_not_sym) {
    // } else if (head == invoke_sym) {
    // } else if (head == call_sym) {
    // } else if (head == foreigncall_sym) {
    // } else if (head == cfunction_sym) {
    // } else if (head == assign_sym) {
    // } else if (head == static_parameter_sym) {
    // } else if (head == method_sym) {
    // } else if (head == const_sym) {
    // } else if (head == new_sym) {
    // } else if (head == splatnew_sym) {
    // } else if (head == exc_sym) {
    // } else if (head == copyast_sym) {
    // } else if (head == loopinfo_sym) {
    // } else if (head == goto_ifnot_sym || head == leave_sym || head == coverageeffect_sym
    //            || head == pop_exception_sym || head == enter_sym || head == inbounds_sym
    //            || head == aliasscope_sym || head == popaliasscope_sym) {
    // } else if (head == boundscheck_sym) {
    // } else if (head == gc_preserve_begin_sym) {
    // } else if (head == gc_preserve_end_sym) {
    // }

    if (head == invoke_sym) {
        // first argument is the `MethodInstance`, second argument is the function
        assert(jl_is_method_instance(args[0]));
        jl_method_instance_t *mi = (jl_method_instance_t*)args[0];
        mlir::Value callee = emit_value(ctx, loc, args[1]);

        // arguments to the `MethodInstance` start from the 3rd argument
        std::vector<mlir::Value> arguments;
        for (unsigned i = 2; i < nargs; ++i) {
            arguments.push_back(emit_value(ctx, loc, args[i]));
        }
        InvokeOp op = ctx.builder.create<InvokeOp>(loc, mi, callee, arguments, type);
        return op.getResult();

    } else if (head == call_sym) {
        mlir::Value callee = emit_value(ctx, loc, args[0]);
        std::vector<mlir::Value> arguments;
        for (unsigned i = 1; i < nargs; ++i) {
            arguments.push_back(emit_value(ctx, loc, args[i]));
        }
        auto op = ctx.builder.create<CallOp>(loc, callee, arguments, type);
        return op.getResult();

    } else {
        auto op = ctx.builder.create<UnimplementedOp>(
            loc, (mlir::Type)JuliaType::get(ctx.context, type));
        return op.getResult();
    }
}

void handleLLVMError(llvm::Error error) {
    llvm::handleAllErrors(std::move(error),
                          [](const llvm::ErrorInfoBase &info) {
                              llvm::errs() << "error: ";
                              info.log(llvm::errs());
                              llvm::errs() << "\n";
                          });
}

mlir::FunctionType emit_ftype(jl_mlirctx_t &ctx,
                              jl_value_t *ir_code,
                              jl_value_t *ret_type) {

    jl_array_t *argtypes = (jl_array_t*)jl_get_field(ir_code, "argtypes");
    size_t nargs = jl_array_dim0(argtypes);
    // FIXME: Handle varargs
    std::vector<mlir::Type> args;
    // First argument is the function, can be needed with call overloading and closures
    for (int i = 0; i < (int)nargs; i++) {
        // this assumes that we have `jl_datatype_t`s!
        args.push_back(
            JuliaType::get(ctx.context, (jl_datatype_t*)jl_arrayref(argtypes, i)));
    }
    mlir::Type ret = (mlir::Type) JuliaType::get(ctx.context, (jl_datatype_t*)ret_type);
    return ctx.builder.getFunctionType(args, llvm::makeArrayRef(ret));
}

mlir::FuncOp emit_function(jl_mlirctx_t &ctx,
                           jl_value_t *ir_code,
                           mlir::FunctionType ftype,
                           mlir::Type ret,
                           jl_value_t* ret_type,
                           std::string name) {
    jl_value_t *irstream = jl_get_field(ir_code, "stmts");

    // 1. Setup debug-information
    std::vector<mlir::Location> locations;
    // `location_indices` is used to convert statement index to location index
    jl_array_t *location_indices = (jl_array_t*) jl_get_field(irstream, "line");
    {
        jl_array_t *linetable = (jl_array_t*) jl_get_field(ir_code, "linetable");
        size_t nlocs = jl_array_len(linetable);
        for (size_t i = 0; i < nlocs; i++) {
            // LineInfoNode(mod::Module, method::Any, file::Symbol, line::Int, inlined_at::Int)
            jl_value_t *locinfo = jl_array_ptr_ref(linetable, i);
            assert(jl_typeis(locinfo, jl_lineinfonode_type));
            jl_module_t *module = (jl_module_t*)jl_fieldref_noalloc(locinfo, 0);
            jl_value_t *method = jl_fieldref_noalloc(locinfo, 1);
            jl_sym_t *filesym = (jl_sym_t*)jl_fieldref_noalloc(locinfo, 2);
            size_t line = jl_unbox_long(jl_fieldref(locinfo, 3));
            size_t inlined_at = jl_unbox_long(jl_fieldref(locinfo, 4));

            if (jl_is_method_instance(method))
                method = ((jl_method_instance_t*)method)->def.value;
            llvm::StringRef file = jl_symbol_name(filesym);

            if (file.empty())
                file = "<missing>";
            llvm::StringRef fname;
            if (jl_is_method(method))
                method = (jl_value_t*)((jl_method_t*)method)->name;
            if (jl_is_symbol(method))
                fname = jl_symbol_name((jl_sym_t*)method);
            if (fname.empty())
                fname = "macro expansion";
            assert(inlined_at <= i);
            mlir::Location current = mlir::NameLoc::get(mlir::Identifier::get(fname, ctx.context),
                                                        mlir::FileLineColLoc::get(file, line, 0, ctx.context));

            // codegen.cpp uses a better heuristic for now just live with this
            if (inlined_at > 0) {
                current = mlir::CallSiteLoc::get(current, locations[inlined_at-1]);
            }
            locations.push_back(current);
        }
    }

    // Create actual function
    mlir::FuncOp function = mlir::FuncOp::create(locations[0], name, ftype);

    // In MLIR the entry block of the function is special: it must have the same
    // argument list as the function itself.
    auto *entryBlock = function.addEntryBlock();
    ctx.builder.setInsertionPointToStart(entryBlock);

    // 3. Get the number of blocks from the CFG and prepare the labels
    jl_value_t *cfg = jl_get_field(ir_code, "cfg");
    jl_array_t *cfg_blocks = (jl_array_t*) jl_get_field(cfg, "blocks");
    int nblocks = (int) jl_array_len(cfg_blocks);

    // Blocks will be 1-index based and block 0 will be the entry block
    std::vector<mlir::Block*> bbs(nblocks + 1);
    bbs[0] = entryBlock;
    for (int i = 1; i <= nblocks; i++) {
        bbs[i] = function.addBlock();
    }

    // 4. Setup conversion
    jl_array_t *stmts = (jl_array_t*)jl_get_field(irstream, "inst");
    jl_array_t *types = (jl_array_t*)jl_get_field(irstream, "type");
    size_t nstmts = jl_array_dim0(stmts);
    ctx.values.resize(nstmts);
    std::copy(entryBlock->args_begin(), entryBlock->args_end(), std::back_inserter(ctx.arguments));

    // Helper function to emit SSAValue, Arguments, GlobalRefs and Constants
    // Helper function to convert PhiNodes into block arguments
    // current_block and target are 1-indexed
    auto emit_branchargs = [&](int current_block, int target, mlir::Location loc) {
        llvm::SmallVector<mlir::Value, 4> v;

        jl_value_t *range = jl_get_field(jl_arrayref(cfg_blocks, target-1), "stmts");
        int start = jl_unbox_long(jl_get_field(range, "start"))-1;
        int stop  = jl_unbox_long(jl_get_field(range, "stop"))-1;
        for (int i = start; start <= stop; ++i) {
            jl_value_t *stmt = jl_arrayref(stmts, i);
            jl_datatype_t *type = (jl_datatype_t*)jl_arrayref(types, i);
            if (jl_is_phinode(stmt)) {
                jl_array_t *edges  = (jl_array_t*)jl_fieldref_noalloc(stmt, 0);
                jl_array_t *values = (jl_array_t*)jl_fieldref_noalloc(stmt, 1);

                int nedges = (int) jl_array_len(edges);
                bool found = false;
                for (int edge = 0; edge < nedges; ++edge) {
                    int frombb = jl_unbox_long(jl_arrayref(edges, edge)); // frombb is 1-indexed
                    if (frombb == current_block) {
                        mlir::Value value =
                            emit_value(ctx, loc, jl_arrayref(values, edge));
                        // if type of block argument is a subtype of the expected
                        // block argument type, use PiOp to widen the value
                        v.push_back(maybe_widen_type(ctx, loc, value, type));
                        found = true;
                    }
                }
                if (!found) {
                    // Julia allows undef PhiNode references to be dropped need to represent them here
                    auto op = ctx.builder.create<UndefOp>(
                        loc, JuliaType::get(ctx.context, type));
                    v.push_back(op.getResult());
                }
            } else {
                // PhiNodes are required to be at the beginning of the basic-blocks
                // so as soon as we find a non-PhiNode we can stop our search.
                break;
            }
        }
        return v;
    };

    // Insert a goto node from the entry block to Julia's first block
    int current_block = 1;
    ctx.builder.create<GotoOp>(mlir::UnknownLoc::get(ctx.context), bbs[current_block], emit_branchargs(0, current_block, locations[0]));
    ctx.builder.setInsertionPointToStart(bbs[current_block]);

    // Process stmts in order
    for (int i = 0; i < (int)nstmts; i++) {
        assert(current_block <= nblocks);
        // XXX: what is jl_array_ptr_ref
        jl_value_t *stmt = jl_arrayref(stmts, i);
        jl_datatype_t *type = (jl_datatype_t*)jl_arrayref(types, i);
        int linetable_index = jl_unbox_int32(jl_arrayref(location_indices, i)); // linetable_index is 1-indexed
        mlir::Location loc = (linetable_index == 0) ?
            mlir::UnknownLoc::get(ctx.context) : locations[linetable_index-1];

        bool is_terminator = false;

        if (jl_isa(stmt, return_node_type)) {
            jl_value_t *ret_val = jl_get_field(stmt, "val");
            Value value;
            if (ret_val) {
                // if type of return value is a subtype of expected return type,
                // use PiOp to widen the value
                value = maybe_widen_type(
                    ctx, loc,
                    emit_value(ctx, loc, ret_val),
                    (jl_datatype_t*)ret_type);
            } else {
                // unreachable terminator, so return undef
                value = ctx.builder.create<UndefOp>(loc, ret);
            }
            ctx.builder.create<ReturnOp>(loc, value);
            is_terminator = true;

        } else if (jl_is_gotonode(stmt)) {
            int label = jl_gotonode_label(stmt);
            assert(label <= nblocks);
            ctx.builder.create<GotoOp>(loc, bbs[label], emit_branchargs(current_block, label, loc));
            is_terminator = true;

        } else if (jl_isa(stmt, gotoifnot_type)) {
            mlir::Value cond = emit_value(ctx, loc, jl_get_field(stmt, "cond"));
            int dest = jl_unbox_long(jl_get_field(stmt, "dest"));
            assert(dest <= nblocks);
            assert(current_block + 1 <= nblocks);
            ctx.builder.create<GotoIfNotOp>(
                loc, cond,
                bbs[dest], emit_branchargs(current_block, dest, loc),
                bbs[current_block + 1], emit_branchargs(current_block, current_block + 1, loc)); // Implicit fallthrough
            is_terminator = true;

        } else if (jl_is_phinode(stmt)) {
            // add argument slot to current_block
            auto arg = bbs[current_block]->addArgument((mlir::Type) JuliaType::get(ctx.context, type));
            // add argument reference to values
            ctx.values[i] = arg;

        } else if (jl_is_pinode(stmt)) {
            jl_value_t *val = jl_get_field(stmt, "val");
            assert(type == (jl_datatype_t*)jl_get_field(stmt, "typ"));
            auto op = ctx.builder.create<PiOp>(loc, emit_value(ctx, loc, val), type);
            ctx.values[i] = op.getResult();

        } else if (jl_is_nothing(stmt)) {
            // ignore dead code

        } else if (jl_is_expr(stmt)) {
            ctx.values[i] = emit_expr(ctx, loc, (jl_expr_t*)stmt, type);

        } else {
            ctx.values[i] = emit_value(ctx, loc, stmt, type);
        }

        // handle implicit fallthrough
        if (!is_terminator) {
            jl_value_t *range = jl_get_field(
                jl_arrayref(cfg_blocks, current_block-1), "stmts");
            int stop  = jl_unbox_long(jl_get_field(range, "stop"))-1;
            if (i == stop) {
                assert(current_block + 1 <= nblocks);
                ctx.builder.create<GotoOp>(
                    loc, bbs[current_block + 1],
                    emit_branchargs(current_block, current_block + 1, loc));
                is_terminator = true;
            }
        }

        if (is_terminator) {
            current_block += 1;
            if (current_block <= nblocks)
                ctx.builder.setInsertionPointToStart(bbs[current_block]);
        }
    }
    return function;
}

extern "C" {

enum DumpOption {
    // DUMP_IRCODE       = 0,
    DUMP_TRANSLATED      = 1,
    DUMP_CANONICALIZED   = 2,
    DUMP_LOWERED_TO_STD  = 4,
    DUMP_LOWERED_TO_LLVM = 8,
};

ExecutionEngineFPtrResult brutus_codegen(jl_value_t *methods,
                                         jl_method_instance_t *entry_mi,
                                         char emit_fptr,
                                         char dump_flags) {
    mlir::MLIRContext context;
    jl_mlirctx_t ctx(&context);

    jl_value_t *entry = jl_call2(getindex_func, methods, (jl_value_t*)entry_mi);
    jl_value_t *ir_code = jl_fieldref(entry, 0);
    jl_value_t *ret_type = jl_fieldref(entry, 1);

    // 1. Create MLIR builder and module
    ModuleOp module = ModuleOp::create(ctx.builder.getUnknownLoc());

    // 2. Function prototype
    mlir::FunctionType ftype = emit_ftype(ctx, ir_code, ret_type);
    // TODO: Fixup singular return
    mlir::Type ret = ftype.getResult(0);

    // 3. Emit function
    std::string name = mi_name(entry_mi);
    mlir::FuncOp function = emit_function(ctx, ir_code, ftype, ret, ret_type, name);
    module.push_back(function);

    if (dump_flags & DUMP_TRANSLATED) {
        llvm::outs() << "after translating to MLIR in JLIR dialect:";
        module.dump();
        llvm::outs() << "\n\n";
    }

    // Lastly verify module
    if (failed(mlir::verify(module))) {
        module.emitError("module verification failed");
        return nullptr;
    }

    // canonicalize

    mlir::PassManager canonicalizePM(&context);
    // Apply any generic pass manager command line options and run the
    // pipeline.
    // FIXME: The next line currently seqfaults
    // applyPassManagerCLOptions(canonicalizePM);
    mlir::OpPassManager &canonicalizeOpPM = canonicalizePM.nest<mlir::FuncOp>();
    canonicalizeOpPM.addPass(mlir::createCanonicalizerPass());
    canonicalizeOpPM.addPass(mlir::createCSEPass());
    LogicalResult canonicalizeResult = canonicalizePM.run(module);;

    if (dump_flags & DUMP_CANONICALIZED) {
        llvm::outs() << "after canonicalizing:";
        module.dump();
        llvm::outs() << "\n\n";
    }

    if (mlir::failed(canonicalizeResult)) {
        module.emitError("module canonicalization failed");
        return nullptr;
    }

    // lower to Standard dialect

    // llvm::DebugFlag = true;
    mlir::PassManager loweringToStdPM(&context);
    loweringToStdPM.addPass(createJLIRToStandardLoweringPass());
    // canonicalize to remove redundant `ConvertStdOp`s
    loweringToStdPM.addPass(mlir::createCanonicalizerPass());
    loweringToStdPM.addPass(mlir::createCSEPass());
    LogicalResult loweringToStdResult = loweringToStdPM.run(module);

    if (dump_flags & DUMP_LOWERED_TO_STD) {
        llvm::outs() << "after lowering to Standard dialect:\n";
        module.dump();
        llvm::outs() << "\n\n";
    }

    if (mlir::failed(loweringToStdResult)) {
        module.emitError("lowering to Standard dialect failed");
        return nullptr;
    }

    // lower to LLVM dialect

    mlir::PassManager loweringToLLVMPM(&context);
    loweringToLLVMPM.addPass(createJLIRToLLVMLoweringPass());
    loweringToLLVMPM.addPass(mlir::createCanonicalizerPass());
    loweringToLLVMPM.addPass(mlir::createCSEPass());
    LogicalResult loweringToLLVMResult = loweringToLLVMPM.run(module);

    if (dump_flags & DUMP_LOWERED_TO_LLVM) {
        llvm::outs() << "after lowering to LLVM dialect:";
        module.dump();
        llvm::outs() << "\n\n";
    }

    if (mlir::failed(loweringToLLVMResult)) {
        module.emitError("lowering to LLVM dialect failed");
        return nullptr;
    }

    // Lastly verify module
    if (failed(mlir::verify(module))) {
        module.emitError("module verification failed");
        return nullptr;
    }

    if (!emit_fptr) {
        return nullptr;
    }

    // see mlir/lib/Support/JitRunner.cpp

    Optional<llvm::CodeGenOpt::Level> jitCodeGenOptLevel =
        llvm::CodeGenOpt::Aggressive;

    auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!tmBuilderOrError) {
        llvm::errs() << "failed to create a JITTargetMachineBuilder for the host\n";
        return nullptr;
    }
    auto tmOrError = tmBuilderOrError->createTargetMachine();
    if (!tmOrError) {
        llvm::errs() << "failed to create a TargetMachine for the host\n";
        return nullptr;
    }

    auto transformer = makeLLVMPassesTransformer(None, 3, tmOrError->get());

    auto expectedEngine = ExecutionEngine::create(
        module, transformer, jitCodeGenOptLevel);
    if (!expectedEngine) {
        handleLLVMError(expectedEngine.takeError());
        return nullptr;
    }

    ExecutionEngine *engine = expectedEngine.get().release();
    auto expectedFPtr = engine->lookup(name);
    if (!expectedFPtr) {
        handleLLVMError(expectedFPtr.takeError());
        return nullptr;
    }

    return expectedFPtr.get();
}

} // extern "C"
