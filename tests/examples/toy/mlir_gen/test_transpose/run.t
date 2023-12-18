  $ cat code.toy
  def main() {
    var a = transpose([[1, 2], [3, 4]]);
  }
  $ dune exec -- toy -emit mlir -f code.toy
  module {
    toy.func @main() {
      %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf64>
      %1 = toy.transpose(%0 : tensor<2x2xf64>) to tensor<*xf64>
      toy.return
    }
  }
