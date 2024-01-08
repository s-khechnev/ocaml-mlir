  $ cat code.toy
  def main() {
    var a = 1 + 1;
    print(a);
  }
  $ dune exec -- toy -emit mlir-llvm -f code.toy
  module attributes {llvm.data_layout = ""} {
    llvm.func @free(!llvm.ptr<i8>)
    llvm.func @malloc(i64) -> !llvm.ptr<i8>
    llvm.func @printMemrefF64(%arg0: i64, %arg1: !llvm.ptr<i8>) attributes {llvm.emit_c_interface, sym_visibility = "private"} {
      %0 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
      %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i64, ptr<i8>)> 
      %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(i64, ptr<i8>)> 
      %3 = llvm.mlir.constant(1 : index) : i64
      %4 = llvm.alloca %3 x !llvm.struct<(i64, ptr<i8>)> : (i64) -> !llvm.ptr<struct<(i64, ptr<i8>)>>
      llvm.store %2, %4 : !llvm.ptr<struct<(i64, ptr<i8>)>>
      llvm.call @_mlir_ciface_printMemrefF64(%4) : (!llvm.ptr<struct<(i64, ptr<i8>)>>) -> ()
      llvm.return
    }
    llvm.func @_mlir_ciface_printMemrefF64(!llvm.ptr<struct<(i64, ptr<i8>)>>) attributes {llvm.emit_c_interface, sym_visibility = "private"}
    llvm.func @main() attributes {llvm.emit_c_interface} {
      %0 = llvm.mlir.constant(1.000000e+00 : f64) : f64
      %1 = llvm.mlir.constant(1 : index) : i64
      %2 = llvm.mlir.null : !llvm.ptr<f64>
      %3 = llvm.getelementptr %2[1] : (!llvm.ptr<f64>) -> !llvm.ptr<f64>
      %4 = llvm.ptrtoint %3 : !llvm.ptr<f64> to i64
      %5 = llvm.call @malloc(%4) : (i64) -> !llvm.ptr<i8>
      %6 = llvm.bitcast %5 : !llvm.ptr<i8> to !llvm.ptr<f64>
      %7 = llvm.mlir.undef : !llvm.struct<(ptr<f64>, ptr<f64>, i64)>
      %8 = llvm.insertvalue %6, %7[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64)> 
      %9 = llvm.insertvalue %6, %8[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64)> 
      %10 = llvm.mlir.constant(0 : index) : i64
      %11 = llvm.insertvalue %10, %9[2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64)> 
      %12 = llvm.mlir.constant(1 : index) : i64
      %13 = llvm.mlir.null : !llvm.ptr<f64>
      %14 = llvm.getelementptr %13[1] : (!llvm.ptr<f64>) -> !llvm.ptr<f64>
      %15 = llvm.ptrtoint %14 : !llvm.ptr<f64> to i64
      %16 = llvm.call @malloc(%15) : (i64) -> !llvm.ptr<i8>
      %17 = llvm.bitcast %16 : !llvm.ptr<i8> to !llvm.ptr<f64>
      %18 = llvm.mlir.undef : !llvm.struct<(ptr<f64>, ptr<f64>, i64)>
      %19 = llvm.insertvalue %17, %18[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64)> 
      %20 = llvm.insertvalue %17, %19[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64)> 
      %21 = llvm.mlir.constant(0 : index) : i64
      %22 = llvm.insertvalue %21, %20[2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64)> 
      llvm.store %0, %17 : !llvm.ptr<f64>
      %23 = llvm.load %17 : !llvm.ptr<f64>
      %24 = llvm.fadd %23, %23  : f64
      llvm.store %24, %6 : !llvm.ptr<f64>
      %25 = llvm.mlir.constant(1 : index) : i64
      %26 = llvm.alloca %25 x !llvm.struct<(ptr<f64>, ptr<f64>, i64)> : (i64) -> !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64)>>
      llvm.store %11, %26 : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64)>>
      %27 = llvm.bitcast %26 : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64)>> to !llvm.ptr<i8>
      %28 = llvm.mlir.constant(0 : index) : i64
      %29 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
      %30 = llvm.insertvalue %28, %29[0] : !llvm.struct<(i64, ptr<i8>)> 
      %31 = llvm.insertvalue %27, %30[1] : !llvm.struct<(i64, ptr<i8>)> 
      llvm.call @printMemrefF64(%28, %27) : (i64, !llvm.ptr<i8>) -> ()
      %32 = llvm.bitcast %17 : !llvm.ptr<f64> to !llvm.ptr<i8>
      llvm.call @free(%32) : (!llvm.ptr<i8>) -> ()
      %33 = llvm.bitcast %6 : !llvm.ptr<f64> to !llvm.ptr<i8>
      llvm.call @free(%33) : (!llvm.ptr<i8>) -> ()
      llvm.return
    }
    llvm.func @_mlir_ciface_main() attributes {llvm.emit_c_interface} {
      llvm.call @main() : () -> ()
      llvm.return
    }
  }
