Brutus
======
*Et tu?*

Brutus is a research project that uses MLIR to implement code-generation and
optimisations for Julia. 


## Setting it up

```
git clone --recursive https://github.com/JuliaLabs/brutus
mkdir build && cd build
cmake ../llvm-project/llvm -GNinja -DLLVM_ENABLE_PROJECTS="mlir" -DLLVM_TARGETS_TO_BUILD="host;NVPTX" \
      -DLLVM_EXTERNAL_PROJECTS="brutus" -DLLVM_EXTERNAL_BRUTUS_SOURCE_DIR=".."
ninja check-brutus
```