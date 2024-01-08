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
      %3 = llvm.mlir.constant(4.000000e+00 : f64) : f64
      %4 = llvm.mlir.constant(5.000000e+00 : f64) : f64
      %5 = llvm.mlir.constant(6.000000e+00 : f64) : f64
      %6 = llvm.mlir.constant(3 : index) : i64
      %7 = llvm.mlir.constant(2 : index) : i64
      %8 = llvm.mlir.constant(1 : index) : i64
      %9 = llvm.mlir.constant(6 : index) : i64
      %10 = llvm.mlir.null : !llvm.ptr<f64>
      %11 = llvm.getelementptr %10[6] : (!llvm.ptr<f64>) -> !llvm.ptr<f64>
      %12 = llvm.ptrtoint %11 : !llvm.ptr<f64> to i64
      %13 = llvm.call @malloc(%12) : (i64) -> !llvm.ptr<i8>
      %14 = llvm.bitcast %13 : !llvm.ptr<i8> to !llvm.ptr<f64>
      %15 = llvm.mlir.undef : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
      %16 = llvm.insertvalue %14, %15[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
      %17 = llvm.insertvalue %14, %16[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
      %18 = llvm.mlir.constant(0 : index) : i64
      %19 = llvm.insertvalue %18, %17[2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
      %20 = llvm.insertvalue %6, %19[3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
      %21 = llvm.insertvalue %7, %20[3, 1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
      %22 = llvm.insertvalue %7, %21[4, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
      %23 = llvm.insertvalue %8, %22[4, 1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
      %24 = llvm.mlir.constant(3 : index) : i64
      %25 = llvm.mlir.constant(2 : index) : i64
      %26 = llvm.mlir.constant(1 : index) : i64
      %27 = llvm.mlir.constant(6 : index) : i64
      %28 = llvm.mlir.null : !llvm.ptr<f64>
      %29 = llvm.getelementptr %28[6] : (!llvm.ptr<f64>) -> !llvm.ptr<f64>
      %30 = llvm.ptrtoint %29 : !llvm.ptr<f64> to i64
      %31 = llvm.call @malloc(%30) : (i64) -> !llvm.ptr<i8>
      %32 = llvm.bitcast %31 : !llvm.ptr<i8> to !llvm.ptr<f64>
      %33 = llvm.mlir.undef : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
      %34 = llvm.insertvalue %32, %33[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
      %35 = llvm.insertvalue %32, %34[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
      %36 = llvm.mlir.constant(0 : index) : i64
      %37 = llvm.insertvalue %36, %35[2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
      %38 = llvm.insertvalue %24, %37[3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
      %39 = llvm.insertvalue %25, %38[3, 1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
      %40 = llvm.insertvalue %25, %39[4, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
      %41 = llvm.insertvalue %26, %40[4, 1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
      %42 = llvm.mlir.constant(2 : index) : i64
      %43 = llvm.mlir.constant(3 : index) : i64
      %44 = llvm.mlir.constant(1 : index) : i64
      %45 = llvm.mlir.constant(6 : index) : i64
      %46 = llvm.mlir.null : !llvm.ptr<f64>
      %47 = llvm.getelementptr %46[6] : (!llvm.ptr<f64>) -> !llvm.ptr<f64>
      %48 = llvm.ptrtoint %47 : !llvm.ptr<f64> to i64
      %49 = llvm.call @malloc(%48) : (i64) -> !llvm.ptr<i8>
      %50 = llvm.bitcast %49 : !llvm.ptr<i8> to !llvm.ptr<f64>
      %51 = llvm.mlir.undef : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>
      %52 = llvm.insertvalue %50, %51[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
      %53 = llvm.insertvalue %50, %52[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
      %54 = llvm.mlir.constant(0 : index) : i64
      %55 = llvm.insertvalue %54, %53[2] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
      %56 = llvm.insertvalue %42, %55[3, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
      %57 = llvm.insertvalue %43, %56[3, 1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
      %58 = llvm.insertvalue %43, %57[4, 0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
      %59 = llvm.insertvalue %44, %58[4, 1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
      %60 = llvm.mlir.constant(1 : index) : i64
      %61 = llvm.mlir.constant(2 : index) : i64
      %62 = llvm.mlir.constant(3 : index) : i64
      %63 = llvm.mul %60, %62  : i64
      %64 = llvm.add %63, %61  : i64
      %65 = llvm.getelementptr %50[%64] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
      llvm.store %5, %65 : !llvm.ptr<f64>
      %66 = llvm.mlir.constant(1 : index) : i64
      %67 = llvm.mlir.constant(1 : index) : i64
      %68 = llvm.mlir.constant(3 : index) : i64
      %69 = llvm.mul %66, %68  : i64
      %70 = llvm.add %69, %67  : i64
      %71 = llvm.getelementptr %50[%70] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
      llvm.store %4, %71 : !llvm.ptr<f64>
      %72 = llvm.mlir.constant(1 : index) : i64
      %73 = llvm.mlir.constant(0 : index) : i64
      %74 = llvm.mlir.constant(3 : index) : i64
      %75 = llvm.mul %72, %74  : i64
      %76 = llvm.add %75, %73  : i64
      %77 = llvm.getelementptr %50[%76] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
      llvm.store %3, %77 : !llvm.ptr<f64>
      %78 = llvm.mlir.constant(0 : index) : i64
      %79 = llvm.mlir.constant(2 : index) : i64
      %80 = llvm.mlir.constant(3 : index) : i64
      %81 = llvm.mul %78, %80  : i64
      %82 = llvm.add %81, %79  : i64
      %83 = llvm.getelementptr %50[%82] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
      llvm.store %2, %83 : !llvm.ptr<f64>
      %84 = llvm.mlir.constant(0 : index) : i64
      %85 = llvm.mlir.constant(1 : index) : i64
      %86 = llvm.mlir.constant(3 : index) : i64
      %87 = llvm.mul %84, %86  : i64
      %88 = llvm.add %87, %85  : i64
      %89 = llvm.getelementptr %50[%88] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
      llvm.store %1, %89 : !llvm.ptr<f64>
      %90 = llvm.mlir.constant(0 : index) : i64
      %91 = llvm.mlir.constant(0 : index) : i64
      %92 = llvm.mlir.constant(3 : index) : i64
      %93 = llvm.mul %90, %92  : i64
      %94 = llvm.add %93, %91  : i64
      %95 = llvm.getelementptr %50[%94] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
      llvm.store %0, %95 : !llvm.ptr<f64>
      %96 = llvm.mlir.constant(0 : index) : i64
      %97 = llvm.mlir.constant(3 : index) : i64
      %98 = llvm.mlir.constant(1 : index) : i64
      llvm.br ^bb1(%96 : i64)
    ^bb1(%99: i64):  // 2 preds: ^bb0, ^bb5
      %100 = llvm.icmp "slt" %99, %97 : i64
      llvm.cond_br %100, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      %101 = llvm.mlir.constant(0 : index) : i64
      %102 = llvm.mlir.constant(2 : index) : i64
      %103 = llvm.mlir.constant(1 : index) : i64
      llvm.br ^bb3(%101 : i64)
    ^bb3(%104: i64):  // 2 preds: ^bb2, ^bb4
      %105 = llvm.icmp "slt" %104, %102 : i64
      llvm.cond_br %105, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      %106 = llvm.mlir.constant(3 : index) : i64
      %107 = llvm.mul %104, %106  : i64
      %108 = llvm.add %107, %99  : i64
      %109 = llvm.getelementptr %50[%108] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
      %110 = llvm.load %109 : !llvm.ptr<f64>
      %111 = llvm.mlir.constant(2 : index) : i64
      %112 = llvm.mul %99, %111  : i64
      %113 = llvm.add %112, %104  : i64
      %114 = llvm.getelementptr %32[%113] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
      llvm.store %110, %114 : !llvm.ptr<f64>
      %115 = llvm.add %104, %103  : i64
      llvm.br ^bb3(%115 : i64)
    ^bb5:  // pred: ^bb3
      %116 = llvm.add %99, %98  : i64
      llvm.br ^bb1(%116 : i64)
    ^bb6:  // pred: ^bb1
      %117 = llvm.mlir.constant(0 : index) : i64
      %118 = llvm.mlir.constant(3 : index) : i64
      %119 = llvm.mlir.constant(1 : index) : i64
      llvm.br ^bb7(%117 : i64)
    ^bb7(%120: i64):  // 2 preds: ^bb6, ^bb11
      %121 = llvm.icmp "slt" %120, %118 : i64
      llvm.cond_br %121, ^bb8, ^bb12
    ^bb8:  // pred: ^bb7
      %122 = llvm.mlir.constant(0 : index) : i64
      %123 = llvm.mlir.constant(2 : index) : i64
      %124 = llvm.mlir.constant(1 : index) : i64
      llvm.br ^bb9(%122 : i64)
    ^bb9(%125: i64):  // 2 preds: ^bb8, ^bb10
      %126 = llvm.icmp "slt" %125, %123 : i64
      llvm.cond_br %126, ^bb10, ^bb11
    ^bb10:  // pred: ^bb9
      %127 = llvm.mlir.constant(2 : index) : i64
      %128 = llvm.mul %120, %127  : i64
      %129 = llvm.add %128, %125  : i64
      %130 = llvm.getelementptr %32[%129] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
      %131 = llvm.load %130 : !llvm.ptr<f64>
      %132 = llvm.fmul %131, %131  : f64
      %133 = llvm.mlir.constant(2 : index) : i64
      %134 = llvm.mul %120, %133  : i64
      %135 = llvm.add %134, %125  : i64
      %136 = llvm.getelementptr %14[%135] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
      llvm.store %132, %136 : !llvm.ptr<f64>
      %137 = llvm.add %125, %124  : i64
      llvm.br ^bb9(%137 : i64)
    ^bb11:  // pred: ^bb9
      %138 = llvm.add %120, %119  : i64
      llvm.br ^bb7(%138 : i64)
    ^bb12:  // pred: ^bb7
      %139 = llvm.mlir.constant(1 : index) : i64
      %140 = llvm.alloca %139 x !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> : (i64) -> !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>>
      llvm.store %23, %140 : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>>
      %141 = llvm.bitcast %140 : !llvm.ptr<struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)>> to !llvm.ptr<i8>
      %142 = llvm.mlir.constant(2 : index) : i64
      %143 = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
      %144 = llvm.insertvalue %142, %143[0] : !llvm.struct<(i64, ptr<i8>)> 
      %145 = llvm.insertvalue %141, %144[1] : !llvm.struct<(i64, ptr<i8>)> 
      llvm.call @printMemrefF64(%142, %141) : (i64, !llvm.ptr<i8>) -> ()
      %146 = llvm.bitcast %50 : !llvm.ptr<f64> to !llvm.ptr<i8>
      llvm.call @free(%146) : (!llvm.ptr<i8>) -> ()
      %147 = llvm.bitcast %32 : !llvm.ptr<f64> to !llvm.ptr<i8>
      llvm.call @free(%147) : (!llvm.ptr<i8>) -> ()
      %148 = llvm.bitcast %14 : !llvm.ptr<f64> to !llvm.ptr<i8>
      llvm.call @free(%148) : (!llvm.ptr<i8>) -> ()
      llvm.return
    }
    llvm.func @_mlir_ciface_main() attributes {llvm.emit_c_interface} {
      llvm.call @main() : () -> ()
      llvm.return
    }
  }
