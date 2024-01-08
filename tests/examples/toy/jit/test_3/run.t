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
  $ dune exec -- toy -emit jit -f code.toy | tail -n +2
  [[1,   16], 
   [4,   25], 
   [9,   36]]
