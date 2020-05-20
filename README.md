Brutus
======
*Et tu?*

Brutus is a research project that uses MLIR to implement code-generation and
optimisations for Julia.


## Setting it up

```
# Build Julia with LLVM 11 and MLIR
git clone https://github.com/yhls/julia
cd julia
git checkout yhls/llvm-11-again
make -j `nproc` \
    USE_BINARYBUILDER_LLVM=0 \
    LLVM_VER=svn \
    LLVM_DEBUG=0 \
    USE_MLIR=1 \
    LLVM_GIT_VER="587f81f54a3abab88fe9be5f113c74fc12655ee0"
cd ..

# Build Brutus
git clone https://github.com/JuliaLabs/brutus
cd brutus && mkdir build && cd build
cmake .. -GNinja \
    -DMLIR_DIR="../../julia/usr/lib/cmake/mlir" \
    -DLLVM_EXTERNAL_LIT="../../julia/usr/tools/lit/lit.py" \
    -DJulia_EXECUTABLE="../../julia/julia" \
    -DCMAKE_BUILD_TYPE=Release
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:`pwd`/../../julia/julia --project=../Brutus -e 'using Pkg; pkg"instantiate"; pkg"precompile"'
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
