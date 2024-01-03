  $ cat code.toy
  def main() {
    var a = [1, 2, 3];
    var b = [1, 2, 3];
    print(a + b);
  }
  $ dune exec -- toy -emit mlir -f code.toy -opt
  module {
    toy.func @main() {
      %0 = toy.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf64>
      %1 = toy.add %0, %0 : tensor<3xf64>
      toy.print %1 : tensor<3xf64>
      toy.return
    }
  }
