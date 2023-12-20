  $ cat code.toy
  def main() {
    print([[1, 2], [3, 4]]);
  }
  $ dune exec -- toy -emit jit -f code.toy
  1.000000 2.000000 
  3.000000 4.000000 
