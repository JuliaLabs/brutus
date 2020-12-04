Brutus
======
*Et tu?*

Brutus is a research project that uses MLIR to implement code-generation and
optimisations for Julia.


## Setting it up

```shell
#!/bin/bash

JULIA_COMMIT_HEAD=e3da30c21747f5dbd9c9c59a825bf3435e32aad2
LLVM_COMMIT_HEAD=094e9f4779eb9b5c6a49014f2f80b8cbb833572f
DEV_DIR=brutus_dev
DEV_PATH=$(pwd)/$DEV_DIR
JULIA_PATH="$DEV_PATH/julia"

# Create development dir.
mkdir $DEV_DIR
cd $DEV_DIR

# Build a development version of Julia with LLVM 12 and MLIR.
git clone https://github.com/JuliaLang/julia
cd julia
git checkout $JULIA_COMMIT_HEAD
make -j `nproc` \
    USE_BINARYBUILDER_LLVM=0 \
    LLVM_VER=svn \
    LLVM_DEBUG=2 \
    USE_MLIR=1 \
    LLVM_GIT_VER="$LLVM_COMMIT_HEAD"
cd ..

# Build brutus.
git clone https://github.com/JuliaLabs/brutus
cd brutus
mkdir build
cd build
cmake .. -GNinja \
    -DMLIR_DIR="${JULIA_PATH}/usr/lib/cmake/mlir" \
    -DLLVM_EXTERNAL_LIT="${JULIA_PATH}/usr/tools/lit/lit.py" \
    -DJulia_EXECUTABLE="${JULIA_PATH}/julia" \
    -DLLVM_BUILD_LIBRARY_DIR="${JULIA_PATH}/usr/tools" \
    -DCMAKE_BUILD_TYPE=Release

# Test brutus.
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(DEV_PATH)/julia/julia --project=../Brutus -e 'using Pkg; pkg"instantiate"; pkg"precompile"'
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
