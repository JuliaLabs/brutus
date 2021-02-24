_Brutus_
======

<figure style="text-align:center;">
<p align="center">
<img height="300px" src="zonkey.png"/>
</p>
</figure>

> *Et tu, Brute?*

`Brutus` is a research project that uses MLIR to implement code-generation and optimisations for Julia.

## Setting it up

`Brutus` currently requires that you create a non-standard build of Julia.

```
git clone https://github.com/JuliaLabs/brutus
cd brutus
git clone https://github.com/JuliaLang/julia
cd julia
git checkout a58bdd90101796eb0ec761a7a8e5103bd96c2d13
make -j `nproc` \
    USE_BINARYBUILDER_LLVM=0 \
    LLVM_VER=svn \
    LLVM_DEBUG=0 \
    USE_MLIR=1 \
    LLVM_GIT_VER="8364f5369eeeb2da8db2bae7716c549930d8df93" 
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
