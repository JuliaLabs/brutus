# Invoke with
# `julia --project=deps deps/build_local.jl [build_dir]`

Brutus = Base.UUID("61eb1bfa-7361-4325-ad38-22787b887f55")

using Pkg, Scratch, Preferences, Libdl, CMake_jll

# 1. Ensure that an appropriate LLVM_full_jll is installed
Pkg.activate(; temp=true)
llvm_assertions = try
    cglobal((:_ZN4llvm24DisableABIBreakingChecksE, Base.libllvm_path()), Cvoid)
    false
catch
    true
end
LLVM = if llvm_assertions
    Pkg.add(name="LLVM_full_assert_jll", version=Base.libllvm_version)
    using LLVM_full_assert_jll
    LLVM_full_assert_jll
else
    Pkg.add(name="LLVM_full_jll", version=Base.libllvm_version)
    using LLVM_full_jll
    LLVM_full_jll
end
MLIR_DIR = joinpath(LLVM.artifact_dir, "lib", "cmake", "mlir")

# 2. Get a scratch directory
if length(ARGS) == 0
    build_dir = get_scratch!(Brutus, "build")
else
    build_dir = only(ARGS)
end
isdir(build_dir) && rm(build_dir; recursive=true)
source_dir = dirname(@__DIR__)
julia=joinpath(Sys.BINDIR::String, Base.julia_exename())

lit = joinpath(LLVM.artifact_dir, "tools", "lit", "lit.py")
mlir_tblgen = joinpath(LLVM.artifact_dir, "tools", "mlir-tblgen")

# Build!
@info "Building" source_dir build_dir MLIR_DIR julia
cmake() do cmake
    run(`$cmake -DMLIR_DIR=$(MLIR_DIR)
                -DJulia_EXECUTABLE=$(julia)
                -DLLVM_EXTERNAL_LIT=$(lit)
                -DMLIR_TABLEGEN_EXE=$(mlir_tblgen)
                -B$(build_dir) -S$(source_dir)`)
    run(`$cmake --build $(build_dir) --parallel $(Sys.CPU_THREADS)`)
end

# FIXME: Discover built libraries
built_libs = filter(readdir(joinpath(scratch_dir, "Enzyme"))) do file
    endswith(file, ".$(Libdl.dlext)") && startswith(file, "lib")
end
lib_path = joinpath(scratch_dir, "Enzyme", only(built_libs))
isfile(lib_path) || error("Could not find library $lib_path in build directory")

# Tell Enzyme_jll to load our library instead of the default artifact one
set_preferences!(
    joinpath(dirname(@__DIR__), "LocalPreferences.toml"),
    "Enzyme_jll",
    "libEnzyme_path" => lib_path;
    force=true,
)
