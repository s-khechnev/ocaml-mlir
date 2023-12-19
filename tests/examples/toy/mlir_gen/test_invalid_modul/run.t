  $ cat code.toy
  def main() {
    var a = 1;
    return;
    var b = 1;
  }
  $ dune exec -- toy -emit mlir -f code.toy
  error: 'toy.return' op must be the last operation in the parent block
  Mlir_gen error: module verification
