  $ cat code.toy
  def main(){
    var a = [1 + 2];
  }
  $ dune exec -- toy -emit ast -f code.toy
  Fatal error: exception Failure(": char ';'")
  Raised at Stdlib.failwith in file "stdlib.ml", line 29, characters 17-33
  Called from Dune__exe__Main in file "examples/toy/main.ml", line 51, characters 19-39
  [2]
