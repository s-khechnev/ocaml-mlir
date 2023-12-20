  $ cat code.toy
  def foo(a) {
    return a * 3;
  }
  
  def main() {
    var a = foo(1, 2);
    print(a);
  }
  $ dune exec -- toy -emit mlir -f code.toy -opt
  Inliner: mismatch number of args for 'foo'
  Some pass fails
