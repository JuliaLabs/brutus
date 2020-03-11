Brutus
======
*Et tu?*

Brutus is a research project that uses MLIR to implement code-generation and
optimisations for Julia. 


## Setting it up

```
# Build Julia with LLVM 11 support
git clone https://github.com/JuliaLang/julia
cd julia
git checkout jn/llvm-11-svn
# Create a file Make.user with the contents:
#   USE_BINARYBUILDER_LLVM=0
#   LLVM_VER=svn
#   LLVM_DEBUG=0
#   LLVM_GIT_VER="e252293d0629066225228596a9ef9d397513eb99"
make -j `nproc`
cd ..

# Build Brutus
git clone --recursive https://github.com/JuliaLabs/brutus
cd brutus && mkdir build && cd build
cmake ../llvm-project/llvm -GNinja \
      -DLLVM_ENABLE_PROJECTS="mlir" \
      -DLLVM_TARGETS_TO_BUILD="host;NVPTX" \
      -DLLVM_EXTERNAL_PROJECTS="brutus" \
      -DLLVM_EXTERNAL_BRUTUS_SOURCE_DIR=".." \
      -DJulia_EXECUTABLE="../../julia/julia"
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:`pwd`/../../julia/julia --project=../Brutus -e 'using Pkg; pkg"instantiate"; pkg"precompile"'
ninja check-brutus
```


### VSCode setting
Using the `Cmake Tools` extension:
```json
{
    "cmake.configureSettings": {
        "LLVM_ENABLE_PROJECTS": "mlir",
        "LLVM_TARGETS_TO_BUILD": "host;NVPTX",
        "LLVM_EXTERNAL_PROJECTS": "brutus",
        "LLVM_EXTERNAL_BRUTUS_SOURCE_DIR": "${workspaceFolder}",
        "Julia_EXECUTABLE": PATH_TO_JULIA_WITH_LLVM_11
    },
    "cmake.sourceDirectory": "${workspaceFolder}/llvm-project/llvm",
    "C_Cpp.default.configurationProvider": "vector-of-bool.cmake-tools"
}
```
