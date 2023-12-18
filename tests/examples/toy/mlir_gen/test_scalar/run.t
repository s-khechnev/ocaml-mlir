  $ cat code.toy
  def main() {
    var a<2, 2> = 5.5;
    print(a);
  }
  $ dune exec -- toy -emit mlir -f code.toy
  module {
    toy.func @main() {
      %0 = toy.constant dense<5.500000e+00> : tensor<f64>
      %1 = toy.reshape(%0 : tensor<f64>) to tensor<2x2xf64>
      toy.print %1 : tensor<2x2xf64>
      toy.return
    }
  }
