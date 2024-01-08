  $ cat code.toy
  def main() {
    var a = [1, 2, 3];
    var b = [3, 2, 1];
    var c = a * b;
    print(c);
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
      %1 = llvm.mlir.constant(2.000000e+00 : f64) : f64
      %2 = llvm.mlir.constant(3.000000e+00 : f64) : f64
      %3 = llvm.mlir.constant(3 : index) : i64
      %4 = llvm.mlir.constant(1 : index) : i64
      %5 = llvm.mlir.null : !llvm.ptr<f64>
      %6 = llvm.getelementptr %5[3] : (!llvm.ptr<f64>) -> !llvm.ptr<f64>
      %7 = llvm.ptrtoint %6 : !llvm.ptr<f64> to i64
      %8 = llvm.call @malloc(%7) : (i64) -> !llvm.ptr<i8>
      %9 = llvm.bitcast %8 : !llvm.ptr<i8> to !llvm.ptr<f64>
      %10 = llvm.mlir.undef : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
      %11 = llvm.insertvalue %9, %10[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
      %12 = llvm.insertvalue %9, %11[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
      %13 = llvm.mlir.constant(0 : index) : i64
      %14 = llvm.insertvalue %13, %12[2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
      %15 = llvm.insertvalue %3, %14[3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
      %16 = llvm.insertvalue %4, %15[4, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
      %17 = llvm.mlir.constant(3 : index) : i64
      %18 = llvm.mlir.constant(1 : index) : i64
      %19 = llvm.mlir.null : !llvm.ptr<f64>
      %20 = llvm.getelementptr %19[3] : (!llvm.ptr<f64>) -> !llvm.ptr<f64>
      %21 = llvm.ptrtoint %20 : !llvm.ptr<f64> to i64
      %22 = llvm.call @malloc(%21) : (i64) -> !llvm.ptr<i8>
      %23 = llvm.bitcast %22 : !llvm.ptr<i8> to !llvm.ptr<f64>
      %24 = llvm.mlir.undef : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
      %25 = llvm.insertvalue %23, %24[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
      %26 = llvm.insertvalue %23, %25[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
      %27 = llvm.mlir.constant(0 : index) : i64
      %28 = llvm.insertvalue %27, %26[2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
      %29 = llvm.insertvalue %17, %28[3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
      %30 = llvm.insertvalue %18, %29[4, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
      %31 = llvm.mlir.constant(3 : index) : i64
      %32 = llvm.mlir.constant(1 : index) : i64
      %33 = llvm.mlir.null : !llvm.ptr<f64>
      %34 = llvm.getelementptr %33[3] : (!llvm.ptr<f64>) -> !llvm.ptr<f64>
      %35 = llvm.ptrtoint %34 : !llvm.ptr<f64> to i64
      %36 = llvm.call @malloc(%35) : (i64) -> !llvm.ptr<i8>
      %37 = llvm.bitcast %36 : !llvm.ptr<i8> to !llvm.ptr<f64>
      %38 = llvm.mlir.undef : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>
      %39 = llvm.insertvalue %37, %38[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
      %40 = llvm.insertvalue %37, %39[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
      %41 = llvm.mlir.constant(0 : index) : i64
      %42 = llvm.insertvalue %41, %40[2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
      %43 = llvm.insertvalue %31, %42[3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
      %44 = llvm.insertvalue %32, %43[4, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
      %45 = llvm.mlir.constant(2 : index) : i64
      %46 = llvm.getelementptr %37[2] : (!llvm.ptr<f64>) -> !llvm.ptr<f64>
      llvm.store %2, %46 : !llvm.ptr<f64>
      %47 = llvm.mlir.constant(1 : index) : i64
      %48 = llvm.getelementptr %37[1] : (!llvm.ptr<f64>) -> !llvm.ptr<f64>
      llvm.store %1, %48 : !llvm.ptr<f64>
      %49 = llvm.mlir.constant(0 : index) : i64
      llvm.store %0, %37 : !llvm.ptr<f64>
      %50 = llvm.mlir.constant(2 : index) : i64
      %51 = llvm.getelementptr %23[2] : (!llvm.ptr<f64>) -> !llvm.ptr<f64>
      llvm.store %0, %51 : !llvm.ptr<f64>
      %52 = llvm.mlir.constant(1 : index) : i64
      %53 = llvm.getelementptr %23[1] : (!llvm.ptr<f64>) -> !llvm.ptr<f64>
      llvm.store %1, %53 : !llvm.ptr<f64>
      %54 = llvm.mlir.constant(0 : index) : i64
      llvm.store %2, %23 : !llvm.ptr<f64>
      %55 = llvm.mlir.constant(0 : index) : i64
      %56 = llvm.mlir.constant(3 : index) : i64
      %57 = llvm.mlir.constant(1 : index) : i64
      llvm.br ^bb1(%55 : i64)
    ^bb1(%58: i64):  // 2 preds: ^bb0, ^bb2
      %59 = llvm.icmp "slt" %58, %56 : i64
      llvm.cond_br %59, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %60 = llvm.getelementptr %37[%58] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
      %61 = llvm.load %60 : !llvm.ptr<f64>
      %62 = llvm.getelementptr %23[%58] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
      %63 = llvm.load %62 : !llvm.ptr<f64>
      %64 = llvm.fmul %61, %63  : f64
      %65 = llvm.getelementptr %9[%58] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
      llvm.store %64, %65 : !llvm.ptr<f64>
      %66 = llvm.add %58, %57  : i64
      llvm.br ^bb1(%66 : i64)
    ^bb3:  // pred: ^bb1
      %67 = llvm.mlir.constant(1 : index) : i64
      %68 = llvm.alloca %67 x !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>
      llvm.store %16, %68 : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>>
      %69 = llvm.bitcast %68 : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)>> to !llvm.ptr<i8>
      %70 = llvm.mlir.constant(1 : index) : i64
      %71 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
      %72 = llvm.insertvalue %70, %71[0] : !llvm.struct<(i64, ptr<i8>)> 
      %73 = llvm.insertvalue %69, %72[1] : !llvm.struct<(i64, ptr<i8>)> 
      llvm.call @printMemrefF64(%70, %69) : (i64, !llvm.ptr<i8>) -> ()
      %74 = llvm.bitcast %37 : !llvm.ptr<f64> to !llvm.ptr<i8>
      llvm.call @free(%74) : (!llvm.ptr<i8>) -> ()
      %75 = llvm.bitcast %23 : !llvm.ptr<f64> to !llvm.ptr<i8>
      llvm.call @free(%75) : (!llvm.ptr<i8>) -> ()
      %76 = llvm.bitcast %9 : !llvm.ptr<f64> to !llvm.ptr<i8>
      llvm.call @free(%76) : (!llvm.ptr<i8>) -> ()
      llvm.return
    }
    llvm.func @_mlir_ciface_main() attributes {llvm.emit_c_interface} {
      llvm.call @main() : () -> ()
      llvm.return
    }
  }
