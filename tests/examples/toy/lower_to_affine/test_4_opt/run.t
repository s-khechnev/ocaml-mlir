  $ cat code.toy
  def main() {
    var a = 1
    var b = 2;
    var c = a * b;
    print(c);
  }
  $ dune exec -- toy -emit mlir-affine -f code.toy -opt
  Fatal error: exception Failure(": char ';'")
  Raised at Stdlib.failwith in file "stdlib.ml", line 29, characters 17-33
  Called from Dune__exe__Main in file "examples/toy/main.ml", line 51, characters 19-39
  [2]
