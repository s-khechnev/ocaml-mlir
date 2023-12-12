[![Build llvm-16.0.6](https://github.com/s-khechnev/ocaml-mlir/actions/workflows/master.yml/badge.svg?branch=llvm-16.0.6)](https://github.com/s-khechnev/ocaml-mlir/actions/workflows/master.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


# OCaml Bindings to MLIR (16.0.6)


## Getting started

1. Required packages
```
llvm-16-dev llvm-16-tools libmlir-16-dev mlir-16-tools
```

2. Current dependencies
```
opam install ctypes ctypes-foreign base stdune ppx_expect ppx_inline_test bisect_ppx
```

3. Build the OCaml bindings

```sh
dune build
```

## API
The entry point to the bindings is `mlir` (see [mlir.mli](src/mlir/mlir.mli)).
It is deliberately low-level, closely matching the MLIR C API.
In the future, we plan to write a higher-level API that wraps around `mlir`. 
This is a first shot at OCaml bindings to the MLIR C API and is likely to undergo major changes in the near future. 

## Toy example

[Here](examples/toy) you can find the implementation of the [Toy tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/).
