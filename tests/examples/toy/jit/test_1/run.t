  $ cat code.toy
  def main() {
    print([[1, 2], [3, 4]]);
  }
  $ dune exec -- toy -emit jit -f code.toy | tail -n +2
  [[1,   2], 
   [3,   4]]
