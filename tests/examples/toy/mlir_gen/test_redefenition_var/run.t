  $ cat code.toy
  def main() {
    var a = 1;
    var a = 2;
  }
  $ dune exec -- toy -emit mlir -f code.toy
  Mlir_gen error: 'a' already defined
