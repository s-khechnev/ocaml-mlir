  $ cat code.toy
  def bar(b) {
    return b * [2, 2];
  }
  
  def foo(a) {
    return a + bar(a);
  }
  
  def main() {
    var a = foo([1, 2]);
    print(a);
  }
  $ dune exec -- toy -emit mlir -f code.toy -opt
  module {
    toy.func @main() {
      %0 = toy.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>
      %1 = toy.constant dense<2.000000e+00> : tensor<2xf64>
      %2 = toy.mul %0, %1 : tensor<2xf64>
      %3 = toy.add %0, %2 : tensor<2xf64>
      toy.print %3 : tensor<2xf64>
      toy.return
    }
  }
