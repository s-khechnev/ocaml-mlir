  $ cat code.toy
  def main() {
    var a = [[1, 2, 3], [4, 5, 6]];
    var b = transpose(a);
    print(b);
  }
  $ dune exec -- toy -emit mlir -f code.toy -opt
  module {
    toy.func @main() {
      %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
      %1 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
      toy.print %1 : tensor<3x2xf64>
      toy.return
    }
  }
