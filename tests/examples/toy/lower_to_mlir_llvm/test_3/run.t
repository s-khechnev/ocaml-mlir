  $ cat code.toy
  def main() {
    var a = [1, 2, 3];
    var b = [3, 2, 1];
    var c = a * b;
    print(c);
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
      %1 = llvm.mlir.constant(2.000000e+00 : f64) : f64
      %2 = llvm.mlir.constant(3.000000e+00 : f64) : f64
      %3 = llvm.mlir.constant(3 : index) : i64
      %4 = llvm.mlir.constant(1 : index) : i64
      %5 = llvm.mlir.null : !llvm.ptr<f64>
      %6 = llvm.getelementptr %5[%3] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
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
      %20 = llvm.getelementptr %19[%17] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
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
      %34 = llvm.getelementptr %33[%31] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
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
      %46 = llvm.extractvalue %44[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
      %47 = llvm.getelementptr %46[%45] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
      llvm.store %2, %47 : !llvm.ptr<f64>
      %48 = llvm.mlir.constant(1 : index) : i64
      %49 = llvm.extractvalue %44[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
      %50 = llvm.getelementptr %49[%48] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
      llvm.store %1, %50 : !llvm.ptr<f64>
      %51 = llvm.mlir.constant(0 : index) : i64
      %52 = llvm.extractvalue %44[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
      %53 = llvm.getelementptr %52[%51] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
      llvm.store %0, %53 : !llvm.ptr<f64>
      %54 = llvm.mlir.constant(2 : index) : i64
      %55 = llvm.extractvalue %30[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
      %56 = llvm.getelementptr %55[%54] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
      llvm.store %0, %56 : !llvm.ptr<f64>
      %57 = llvm.mlir.constant(1 : index) : i64
      %58 = llvm.extractvalue %30[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
      %59 = llvm.getelementptr %58[%57] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
      llvm.store %1, %59 : !llvm.ptr<f64>
      %60 = llvm.mlir.constant(0 : index) : i64
      %61 = llvm.extractvalue %30[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
      %62 = llvm.getelementptr %61[%60] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
      llvm.store %2, %62 : !llvm.ptr<f64>
      %63 = llvm.mlir.constant(0 : index) : i64
      %64 = llvm.mlir.constant(3 : index) : i64
      %65 = llvm.mlir.constant(1 : index) : i64
      llvm.br ^bb1(%63 : i64)
    ^bb1(%66: i64):  // 2 preds: ^bb0, ^bb2
      %67 = llvm.icmp "slt" %66, %64 : i64
      llvm.cond_br %67, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %68 = llvm.extractvalue %44[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
      %69 = llvm.getelementptr %68[%66] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
      %70 = llvm.load %69 : !llvm.ptr<f64>
      %71 = llvm.extractvalue %30[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
      %72 = llvm.getelementptr %71[%66] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
      %73 = llvm.load %72 : !llvm.ptr<f64>
      %74 = llvm.fmul %70, %73  : f64
      %75 = llvm.extractvalue %16[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
      %76 = llvm.getelementptr %75[%66] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
      llvm.store %74, %76 : !llvm.ptr<f64>
      %77 = llvm.add %66, %65  : i64
      llvm.br ^bb1(%77 : i64)
    ^bb3:  // pred: ^bb1
      %78 = llvm.mlir.addressof @frmt_spec : !llvm.ptr<array<4 x i8>>
      %79 = llvm.mlir.constant(0 : index) : i64
      %80 = llvm.getelementptr %78[%79, %79] : (!llvm.ptr<array<4 x i8>>, i64, i64) -> !llvm.ptr<i8>
      %81 = llvm.mlir.addressof @nl : !llvm.ptr<array<2 x i8>>
      %82 = llvm.mlir.constant(0 : index) : i64
      %83 = llvm.getelementptr %81[%82, %82] : (!llvm.ptr<array<2 x i8>>, i64, i64) -> !llvm.ptr<i8>
      %84 = llvm.mlir.constant(0 : index) : i64
      %85 = llvm.mlir.constant(3 : index) : i64
      %86 = llvm.mlir.constant(1 : index) : i64
      llvm.br ^bb4(%84 : i64)
    ^bb4(%87: i64):  // 2 preds: ^bb3, ^bb5
      %88 = llvm.icmp "slt" %87, %85 : i64
      llvm.cond_br %88, ^bb5, ^bb6
    ^bb5:  // pred: ^bb4
      %89 = llvm.extractvalue %16[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
      %90 = llvm.getelementptr %89[%87] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
      %91 = llvm.load %90 : !llvm.ptr<f64>
      %92 = llvm.call @printf(%80, %91) : (!llvm.ptr<i8>, f64) -> i32
      %93 = llvm.add %87, %86  : i64
      llvm.br ^bb4(%93 : i64)
    ^bb6:  // pred: ^bb4
      %94 = llvm.extractvalue %44[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
      %95 = llvm.bitcast %94 : !llvm.ptr<f64> to !llvm.ptr<i8>
      llvm.call @free(%95) : (!llvm.ptr<i8>) -> ()
      %96 = llvm.extractvalue %30[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
      %97 = llvm.bitcast %96 : !llvm.ptr<f64> to !llvm.ptr<i8>
      llvm.call @free(%97) : (!llvm.ptr<i8>) -> ()
      %98 = llvm.extractvalue %16[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<1 x i64>, array<1 x i64>)> 
      %99 = llvm.bitcast %98 : !llvm.ptr<f64> to !llvm.ptr<i8>
      llvm.call @free(%99) : (!llvm.ptr<i8>) -> ()
      llvm.return
    }
    llvm.func @_mlir_ciface_main() attributes {llvm.emit_c_interface} {
      llvm.call @main() : () -> ()
      llvm.return
    }
  }
