  $ cat code.toy
  def bar(a) {
    return a * [3, 4];
  }
  
  def main() {
    var a = foo([1, 2]);
    print(a);
  }
  $ dune exec -- toy -emit mlir -f code.toy -opt
  Inliner: 'foo' function not found
  Some pass fails
