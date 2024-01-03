  $ cat code.toy
  def main() {
    var a = 1;
    var b = 2;
    var c = a * b;
    print(c);
  }
  $ dune exec -- toy -emit mlir-affine -f code.toy -opt
  module {
    func.func @main() attributes {llvm.emit_c_interface} {
      %cst = arith.constant 2.000000e+00 : f64
      %cst_0 = arith.constant 1.000000e+00 : f64
      %alloc = memref.alloc() : memref<f64>
      %0 = arith.mulf %cst_0, %cst : f64
      affine.store %0, %alloc[] : memref<f64>
      toy.print %alloc : memref<f64>
      memref.dealloc %alloc : memref<f64>
      return
    }
  }
