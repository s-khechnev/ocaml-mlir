  $ cat code.toy
  def bar(a) {
    return a * [3, 4];
  }
  
  def foo() {
    var a = bar([1, 2]);
    print(a);
  }
  $ dune exec -- toy -emit mlir -f code.toy -opt
  Inliner: 'main' function not found
  Some pass fails
