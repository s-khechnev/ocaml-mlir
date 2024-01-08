  $ cat code.toy
  def plus_transpose(a, b) {
    return transpose(a) + transpose(b);
  }
  
  def main() {
    var a<6> = [[1, 2, 3], [4, 5, 6]];
    var b = [1, 2, 3, 4, 5, 6];
    var c = plus_transpose(a, b);
    print(c);
  }
  $ dune exec -- toy -emit mlir-affine -f code.toy
  module {
    func.func private @printMemrefF64(memref<*xf64>) attributes {llvm.emit_c_interface}
    func.func @main() attributes {llvm.emit_c_interface} {
      %cst = arith.constant 1.000000e+00 : f64
      %cst_0 = arith.constant 2.000000e+00 : f64
      %cst_1 = arith.constant 3.000000e+00 : f64
      %cst_2 = arith.constant 4.000000e+00 : f64
      %cst_3 = arith.constant 5.000000e+00 : f64
      %cst_4 = arith.constant 6.000000e+00 : f64
      %alloc = memref.alloc() : memref<6xf64>
      %alloc_5 = memref.alloc() : memref<6xf64>
      %alloc_6 = memref.alloc() : memref<6xf64>
      affine.store %cst_4, %alloc_6[5] : memref<6xf64>
      affine.store %cst_3, %alloc_6[4] : memref<6xf64>
      affine.store %cst_2, %alloc_6[3] : memref<6xf64>
      affine.store %cst_1, %alloc_6[2] : memref<6xf64>
      affine.store %cst_0, %alloc_6[1] : memref<6xf64>
      affine.store %cst, %alloc_6[0] : memref<6xf64>
      affine.for %arg0 = 0 to 6 {
        %0 = affine.load %alloc_6[%arg0] : memref<6xf64>
        affine.store %0, %alloc_5[%arg0] : memref<6xf64>
      }
      affine.for %arg0 = 0 to 6 {
        %0 = affine.load %alloc_5[%arg0] : memref<6xf64>
        %1 = arith.addf %0, %0 : f64
        affine.store %1, %alloc[%arg0] : memref<6xf64>
      }
      %cast = memref.cast %alloc : memref<6xf64> to memref<*xf64>
      call @printMemrefF64(%cast) : (memref<*xf64>) -> ()
      memref.dealloc %alloc_6 : memref<6xf64>
      memref.dealloc %alloc_5 : memref<6xf64>
      memref.dealloc %alloc : memref<6xf64>
      return
    }
  }
