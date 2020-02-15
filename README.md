Brutus
======
*Et tu?*

Brutus is a research project that uses MLIR to implement code-generation and
optimisations for Julia. 


## Setting it up

```
git clone https://github.com/llvm/llvm-project
cd llvm-project/llvm/projects
git clone https://github.com/JuliaLabs/brutus
cd ../
mkdir build
cd build
cmake -GNinja .. -DLLVM_ENABLE_PROJECTS="mlir" -DLLVM_TARGETS_TO_BUILD="host;NVPTX"
ninja
```