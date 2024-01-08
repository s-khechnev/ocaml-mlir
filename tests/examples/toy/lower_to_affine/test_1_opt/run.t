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
  $ dune exec -- toy -emit mlir-affine -f code.toy -opt
  module {
    func.func private @printMemrefF64(memref<*xf64>) attributes {llvm.emit_c_interface}
    func.func @main() attributes {llvm.emit_c_interface} {
      %cst = arith.constant 1.000000e+00 : f64
      %cst_0 = arith.constant 2.000000e+00 : f64
      %cst_1 = arith.constant 3.000000e+00 : f64
      %cst_2 = arith.constant 4.000000e+00 : f64
      %cst_3 = arith.constant 5.000000e+00 : f64
      %cst_4 = arith.constant 6.000000e+00 : f64
      %alloc = memref.alloc() : memref<3x2xf64>
      %alloc_5 = memref.alloc() : memref<2x3xf64>
      affine.store %cst_4, %alloc_5[1, 2] : memref<2x3xf64>
      affine.store %cst_3, %alloc_5[1, 1] : memref<2x3xf64>
      affine.store %cst_2, %alloc_5[1, 0] : memref<2x3xf64>
      affine.store %cst_1, %alloc_5[0, 2] : memref<2x3xf64>
      affine.store %cst_0, %alloc_5[0, 1] : memref<2x3xf64>
      affine.store %cst, %alloc_5[0, 0] : memref<2x3xf64>
      affine.for %arg0 = 0 to 3 {
        affine.for %arg1 = 0 to 2 {
          %0 = affine.load %alloc_5[%arg1, %arg0] : memref<2x3xf64>
          %1 = arith.mulf %0, %0 : f64
          affine.store %1, %alloc[%arg0, %arg1] : memref<3x2xf64>
        }
      }
      %cast = memref.cast %alloc : memref<3x2xf64> to memref<*xf64>
      call @printMemrefF64(%cast) : (memref<*xf64>) -> ()
      memref.dealloc %alloc_5 : memref<2x3xf64>
      memref.dealloc %alloc : memref<3x2xf64>
      return
    }
  }
