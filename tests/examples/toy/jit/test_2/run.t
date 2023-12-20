  $ cat code.toy
  def main() {
    print(([2] + [2]) * [2]);
  }
  $ dune exec -- toy -emit jit -f code.toy
  8.000000 
