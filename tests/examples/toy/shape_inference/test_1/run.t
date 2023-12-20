  $ cat code.toy
  def multiply_transpose(a, b) {
    return transpose(a) * transpose(b);
  }
  
  def main() {
    var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
    var b<2, 3> = [1, 2, 3, 4, 5, 6];
    var c = multiply_transpose(a, b);
    print(c);
  }
  $ dune exec -- toy -emit mlir -f code.toy -opt
  module {
    toy.func @main() {
      %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
      %1 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
      %2 = toy.mul %1, %1 : tensor<3x2xf64>
      toy.print %2 : tensor<3x2xf64>
      toy.return
    }
  }
