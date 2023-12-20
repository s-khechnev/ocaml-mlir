  $ cat code.toy
  def plus_transpose(a, b) {
    return transpose(a) + transpose(b);
  }
  
  def main() {
    var a<6> = [[1, 2, 3], [4, 5, 6]];
    var b = [1, 2, 3, 4, 5, 6];
    var c = plus_transpose(a, b);
    print(c);
  }
  $ dune exec -- toy -emit mlir -f code.toy -opt
  module {
    toy.func @main() {
      %0 = toy.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>
      %1 = toy.transpose(%0 : tensor<6xf64>) to tensor<6xf64>
      %2 = toy.add %1, %1 : tensor<6xf64>
      toy.print %2 : tensor<6xf64>
      toy.return
    }
  }
