  $ cat code.toy
  def foo(a, b) {
    return a * b;
  }
  
  def main() {
    var a = foo(1);
    print(a);
  }
  $ dune exec -- toy -emit mlir -f code.toy -opt
  Inliner: mismatch number of args for 'foo'
  Some pass fails
