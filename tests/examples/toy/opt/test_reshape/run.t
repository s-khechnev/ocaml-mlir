  $ cat code.toy
  def main() {
    var a<3, 2> = [[1, 2, 3], [4, 5, 6]];
    print(a);
  }
  $ dune exec -- toy -emit mlir -f code.toy -opt
  module {
    toy.func @main() {
      %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00], [5.000000e+00, 6.000000e+00]]> : tensor<3x2xf64>
      toy.print %0 : tensor<3x2xf64>
      toy.return
    }
  }
