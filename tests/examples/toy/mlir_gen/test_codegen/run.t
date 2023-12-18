  $ cat code.toy
  def multiply_transpose(a, b) {
    return transpose(a) * transpose(b);
  }
  
  def main() {
    var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
    var b<2, 3> = [1, 2, 3, 4, 5, 6];
    var c = multiply_transpose(a, b);
    var d = multiply_transpose(b, a);
    print(d);
  }
  $ dune exec -- toy -emit mlir -f code.toy
  module {
    toy.func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
      %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
      %1 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64>
      %2 = toy.mul %0, %1 : tensor<*xf64>
      toy.return %2 : tensor<*xf64>
    }
    toy.func @main() {
      %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
      %1 = toy.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64>
      %2 = toy.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>
      %3 = toy.reshape(%2 : tensor<6xf64>) to tensor<2x3xf64>
      %4 = toy.generic_call @multiply_transpose(%1, %3) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
      %5 = toy.generic_call @multiply_transpose(%3, %1) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
      toy.print %5 : tensor<*xf64>
      toy.return
    }
  }
