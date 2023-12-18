  $ cat code.toy
  def main() {
    var a = 1;
  }
  $ dune exec -- toy -emit mlir -f code.toy
  module {
    toy.func @main() {
      %0 = toy.constant dense<1.000000e+00> : tensor<f64>
      toy.return
    }
  }
