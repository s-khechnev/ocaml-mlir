  $ cat code.toy
  def foo(a) {
    return a + [3, 4];
  }
  
  def main() {
    var a = foo([1, 2]);
    print(a);
  }
  $ dune exec -- toy -emit mlir -f code.toy -opt
  module {
    toy.func @main() {
      %0 = toy.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>
      %1 = toy.constant dense<[3.000000e+00, 4.000000e+00]> : tensor<2xf64>
      %2 = toy.add %0, %1 : tensor<2xf64>
      toy.print %2 : tensor<2xf64>
      toy.return
    }
  }
