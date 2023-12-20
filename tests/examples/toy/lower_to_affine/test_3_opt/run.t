  $ cat code.toy
  def main() {
    var a = [1, 2, 3];
    var b<3> = [[1, 2, 3]];
    var c = a * b;
    print(c);
  }
  $ dune exec -- toy -emit mlir-affine -f code.toy -opt
  module {
    func.func @main() attributes {llvm.emit_c_interface} {
      %cst = arith.constant 1.000000e+00 : f64
      %cst_0 = arith.constant 2.000000e+00 : f64
      %cst_1 = arith.constant 3.000000e+00 : f64
      %alloc = memref.alloc() : memref<3xf64>
      %alloc_2 = memref.alloc() : memref<3xf64>
      affine.store %cst_1, %alloc_2[2] : memref<3xf64>
      affine.store %cst_0, %alloc_2[1] : memref<3xf64>
      affine.store %cst, %alloc_2[0] : memref<3xf64>
      affine.for %arg0 = 0 to 3 {
        %0 = affine.load %alloc_2[%arg0] : memref<3xf64>
        %1 = arith.mulf %0, %0 : f64
        affine.store %1, %alloc[%arg0] : memref<3xf64>
      }
      toy.print %alloc : memref<3xf64>
      memref.dealloc %alloc_2 : memref<3xf64>
      memref.dealloc %alloc : memref<3xf64>
      return
    }
  }
