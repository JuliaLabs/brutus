_Brutus_
======

> *Et tu, Brute?*

`Brutus` is a research project that uses MLIR to implement code-generation and optimisations for Julia.

## Setting it up

`Brutus` currently requires that you create a non-standard build of Julia.

```
export LLVM_SHA1=4743f8ded72e15f916fa1d4cc198bdfd7bfb2193 # LLVM 13.0.1-0
export JULIA_SHA1=6c16f717f9871401eed9350f36cd84ab51778b72 # Julia 1.8-dev

git clone https://github.com/JuliaLabs/brutus
cd brutus
git clone https://github.com/JuliaLang/julia
cd julia
git checkout ${JULIA_SHA1}
make -j `nproc` \
    USE_BINARYBUILDER_LLVM=0 \
    DEPS_GIT=1 \
    LLVM_DEBUG=0 \
    USE_MLIR=1 \
    LLVM_SHA1="${LLVM_SHA1}"
cd ..
mkdir build && cd build
cmake .. -G Ninja \
    -DMLIR_DIR="$(pwd)/../julia/usr/lib/cmake/mlir" \
    -DLLVM_EXTERNAL_LIT="$(pwd)/../julia/usr/tools/lit/lit.py" \
    -DJulia_EXECUTABLE="$(pwd)/../julia/julia" \
    -DCMAKE_BUILD_TYPE=Release
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:`pwd`/../julia/julia --project=../Brutus -e 'using Pkg; pkg"instantiate"; pkg"precompile"'
ninja check-brutus
```

### VSCode setting

Using the `Cmake Tools` extension, and with the path to Julia as built above in
place of `JULIA_PATH`:
```json
{
    "cmake.configureSettings": {
        "MLIR_DIR": "${JULIA_PATH}/usr/lib/cmake/mlir",
        "LLVM_EXTERNAL_LIT": "${JULIA_PATH}/usr/tools/lit/lit.py",
        "Julia_EXECUTABLE": "${JULIA_PATH}/julia",
        "CMAKE_BUILD_TYPE": "Release"
    },
    "cmake.sourceDirectory": "${workspaceFolder}",
    "C_Cpp.default.configurationProvider": "vector-of-bool.cmake-tools"
}
```
