  $ cat code.toy
  def main() {
    var a = 1;
    var a = 2;
  }
  $ dune exec -- toy -emit mlir -f code.toy
  Fatal error: exception Failure("'a' already defined.")
  Raised at Stdlib.failwith in file "stdlib.ml", line 29, characters 17-33
  Called from Dune__exe__Mlir_gen.mlirgen_func.(fun) in file "examples/toy/mlir_gen.ml", line 153, characters 19-37
  Called from Base__List0.iter in file "src/list0.ml", line 60, characters 4-7
  Called from Dune__exe__Mlir_gen.mlirgen_func in file "examples/toy/mlir_gen.ml", line 152, characters 9-93
  Called from Dune__exe__Mlir_gen.mlirgen.(fun) in file "examples/toy/mlir_gen.ml", line 192, characters 14-28
  Called from Base__List0.iter in file "src/list0.ml", line 60, characters 4-7
  Called from Dune__exe__Mlir_gen.mlirgen in file "examples/toy/mlir_gen.ml", line 191, characters 4-115
  Called from Dune__exe__Main in file "examples/toy/main.ml", line 58, characters 22-44
  [2]
