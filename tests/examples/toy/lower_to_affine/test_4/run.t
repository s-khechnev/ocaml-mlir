  $ cat code.toy
  def main() {
    var a = 1;
    var b = 2;
    var c = a * b;
    print(c);
  }
  $ dune exec -- toy -emit mlir-affine -f code.toy
  module {
    func.func private @printMemrefF64(memref<*xf64>) attributes {llvm.emit_c_interface}
    func.func @main() attributes {llvm.emit_c_interface} {
      %cst = arith.constant 2.000000e+00 : f64
      %cst_0 = arith.constant 1.000000e+00 : f64
      %alloc = memref.alloc() : memref<f64>
      %alloc_1 = memref.alloc() : memref<f64>
      %alloc_2 = memref.alloc() : memref<f64>
      affine.store %cst_0, %alloc_2[] : memref<f64>
      affine.store %cst, %alloc_1[] : memref<f64>
      %0 = affine.load %alloc_2[] : memref<f64>
      %1 = affine.load %alloc_1[] : memref<f64>
      %2 = arith.mulf %0, %1 : f64
      affine.store %2, %alloc[] : memref<f64>
      %cast = memref.cast %alloc : memref<f64> to memref<*xf64>
      call @printMemrefF64(%cast) : (memref<*xf64>) -> ()
      memref.dealloc %alloc_2 : memref<f64>
      memref.dealloc %alloc_1 : memref<f64>
      memref.dealloc %alloc : memref<f64>
      return
    }
  }
