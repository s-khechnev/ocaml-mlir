  $ cat code.toy
  def main() {
    var a = 1 + 1;
    print(a);
  }
  $ dune exec -- toy -emit mlir-llvm -f code.toy
  module {
    llvm.func @free(!llvm.ptr<i8>)
    llvm.mlir.global internal constant @nl("\0A\00") {addr_space = 0 : i32}
    llvm.mlir.global internal constant @frmt_spec("%f \00") {addr_space = 0 : i32}
    llvm.func @printf(!llvm.ptr<i8>, ...) -> i32
    llvm.func @malloc(i64) -> !llvm.ptr<i8>
    llvm.func @main() attributes {llvm.emit_c_interface} {
      %0 = llvm.mlir.constant(1.000000e+00 : f64) : f64
      %1 = llvm.mlir.constant(1 : index) : i64
      %2 = llvm.mlir.null : !llvm.ptr<f64>
      %3 = llvm.getelementptr %2[%1] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
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
      %14 = llvm.getelementptr %13[%12] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
      %15 = llvm.ptrtoint %14 : !llvm.ptr<f64> to i64
      %16 = llvm.call @malloc(%15) : (i64) -> !llvm.ptr<i8>
      %17 = llvm.bitcast %16 : !llvm.ptr<i8> to !llvm.ptr<f64>
      %18 = llvm.mlir.undef : !llvm.struct<(ptr<f64>, ptr<f64>, i64)>
      %19 = llvm.insertvalue %17, %18[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64)> 
      %20 = llvm.insertvalue %17, %19[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64)> 
      %21 = llvm.mlir.constant(0 : index) : i64
      %22 = llvm.insertvalue %21, %20[2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64)> 
      %23 = llvm.extractvalue %22[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64)> 
      llvm.store %0, %23 : !llvm.ptr<f64>
      %24 = llvm.extractvalue %22[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64)> 
      %25 = llvm.load %24 : !llvm.ptr<f64>
      %26 = llvm.fadd %25, %25  : f64
      %27 = llvm.extractvalue %11[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64)> 
      llvm.store %26, %27 : !llvm.ptr<f64>
      %28 = llvm.mlir.addressof @frmt_spec : !llvm.ptr<array<4 x i8>>
      %29 = llvm.mlir.constant(0 : index) : i64
      %30 = llvm.getelementptr %28[%29, %29] : (!llvm.ptr<array<4 x i8>>, i64, i64) -> !llvm.ptr<i8>
      %31 = llvm.mlir.addressof @nl : !llvm.ptr<array<2 x i8>>
      %32 = llvm.mlir.constant(0 : index) : i64
      %33 = llvm.getelementptr %31[%32, %32] : (!llvm.ptr<array<2 x i8>>, i64, i64) -> !llvm.ptr<i8>
      %34 = llvm.extractvalue %11[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64)> 
      %35 = llvm.load %34 : !llvm.ptr<f64>
      %36 = llvm.call @printf(%30, %35) : (!llvm.ptr<i8>, f64) -> i32
      %37 = llvm.extractvalue %22[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64)> 
      %38 = llvm.bitcast %37 : !llvm.ptr<f64> to !llvm.ptr<i8>
      llvm.call @free(%38) : (!llvm.ptr<i8>) -> ()
      %39 = llvm.extractvalue %11[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64)> 
      %40 = llvm.bitcast %39 : !llvm.ptr<f64> to !llvm.ptr<i8>
      llvm.call @free(%40) : (!llvm.ptr<i8>) -> ()
      llvm.return
    }
    llvm.func @_mlir_ciface_main() attributes {llvm.emit_c_interface} {
      llvm.call @main() : () -> ()
      llvm.return
    }
  }
