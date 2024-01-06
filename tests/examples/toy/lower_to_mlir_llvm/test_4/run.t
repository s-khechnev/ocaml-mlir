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
      %3 = llvm.mlir.constant(4.000000e+00 : f64) : f64
      %4 = llvm.mlir.constant(5.000000e+00 : f64) : f64
      %5 = llvm.mlir.constant(6.000000e+00 : f64) : f64
      %6 = llvm.mlir.constant(3 : index) : i64
      %7 = llvm.mlir.constant(2 : index) : i64
      %8 = llvm.mlir.constant(1 : index) : i64
      %9 = llvm.mlir.constant(6 : index) : i64
      %10 = llvm.mlir.null : !llvm.ptr<f64>
      %11 = llvm.getelementptr %10[%9] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
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
      %29 = llvm.getelementptr %28[%27] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
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
      %47 = llvm.getelementptr %46[%45] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
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
      %62 = llvm.extractvalue %59[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
      %63 = llvm.mlir.constant(3 : index) : i64
      %64 = llvm.mul %60, %63  : i64
      %65 = llvm.add %64, %61  : i64
      %66 = llvm.getelementptr %62[%65] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
      llvm.store %5, %66 : !llvm.ptr<f64>
      %67 = llvm.mlir.constant(1 : index) : i64
      %68 = llvm.mlir.constant(1 : index) : i64
      %69 = llvm.extractvalue %59[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
      %70 = llvm.mlir.constant(3 : index) : i64
      %71 = llvm.mul %67, %70  : i64
      %72 = llvm.add %71, %68  : i64
      %73 = llvm.getelementptr %69[%72] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
      llvm.store %4, %73 : !llvm.ptr<f64>
      %74 = llvm.mlir.constant(1 : index) : i64
      %75 = llvm.mlir.constant(0 : index) : i64
      %76 = llvm.extractvalue %59[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
      %77 = llvm.mlir.constant(3 : index) : i64
      %78 = llvm.mul %74, %77  : i64
      %79 = llvm.add %78, %75  : i64
      %80 = llvm.getelementptr %76[%79] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
      llvm.store %3, %80 : !llvm.ptr<f64>
      %81 = llvm.mlir.constant(0 : index) : i64
      %82 = llvm.mlir.constant(2 : index) : i64
      %83 = llvm.extractvalue %59[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
      %84 = llvm.mlir.constant(3 : index) : i64
      %85 = llvm.mul %81, %84  : i64
      %86 = llvm.add %85, %82  : i64
      %87 = llvm.getelementptr %83[%86] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
      llvm.store %2, %87 : !llvm.ptr<f64>
      %88 = llvm.mlir.constant(0 : index) : i64
      %89 = llvm.mlir.constant(1 : index) : i64
      %90 = llvm.extractvalue %59[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
      %91 = llvm.mlir.constant(3 : index) : i64
      %92 = llvm.mul %88, %91  : i64
      %93 = llvm.add %92, %89  : i64
      %94 = llvm.getelementptr %90[%93] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
      llvm.store %1, %94 : !llvm.ptr<f64>
      %95 = llvm.mlir.constant(0 : index) : i64
      %96 = llvm.mlir.constant(0 : index) : i64
      %97 = llvm.extractvalue %59[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
      %98 = llvm.mlir.constant(3 : index) : i64
      %99 = llvm.mul %95, %98  : i64
      %100 = llvm.add %99, %96  : i64
      %101 = llvm.getelementptr %97[%100] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
      llvm.store %0, %101 : !llvm.ptr<f64>
      %102 = llvm.mlir.constant(0 : index) : i64
      %103 = llvm.mlir.constant(3 : index) : i64
      %104 = llvm.mlir.constant(1 : index) : i64
      llvm.br ^bb1(%102 : i64)
    ^bb1(%105: i64):  // 2 preds: ^bb0, ^bb5
      %106 = llvm.icmp "slt" %105, %103 : i64
      llvm.cond_br %106, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      %107 = llvm.mlir.constant(0 : index) : i64
      %108 = llvm.mlir.constant(2 : index) : i64
      %109 = llvm.mlir.constant(1 : index) : i64
      llvm.br ^bb3(%107 : i64)
    ^bb3(%110: i64):  // 2 preds: ^bb2, ^bb4
      %111 = llvm.icmp "slt" %110, %108 : i64
      llvm.cond_br %111, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      %112 = llvm.extractvalue %59[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
      %113 = llvm.mlir.constant(3 : index) : i64
      %114 = llvm.mul %110, %113  : i64
      %115 = llvm.add %114, %105  : i64
      %116 = llvm.getelementptr %112[%115] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
      %117 = llvm.load %116 : !llvm.ptr<f64>
      %118 = llvm.extractvalue %41[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
      %119 = llvm.mlir.constant(2 : index) : i64
      %120 = llvm.mul %105, %119  : i64
      %121 = llvm.add %120, %110  : i64
      %122 = llvm.getelementptr %118[%121] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
      llvm.store %117, %122 : !llvm.ptr<f64>
      %123 = llvm.add %110, %109  : i64
      llvm.br ^bb3(%123 : i64)
    ^bb5:  // pred: ^bb3
      %124 = llvm.add %105, %104  : i64
      llvm.br ^bb1(%124 : i64)
    ^bb6:  // pred: ^bb1
      %125 = llvm.mlir.constant(0 : index) : i64
      %126 = llvm.mlir.constant(3 : index) : i64
      %127 = llvm.mlir.constant(1 : index) : i64
      llvm.br ^bb7(%125 : i64)
    ^bb7(%128: i64):  // 2 preds: ^bb6, ^bb11
      %129 = llvm.icmp "slt" %128, %126 : i64
      llvm.cond_br %129, ^bb8, ^bb12
    ^bb8:  // pred: ^bb7
      %130 = llvm.mlir.constant(0 : index) : i64
      %131 = llvm.mlir.constant(2 : index) : i64
      %132 = llvm.mlir.constant(1 : index) : i64
      llvm.br ^bb9(%130 : i64)
    ^bb9(%133: i64):  // 2 preds: ^bb8, ^bb10
      %134 = llvm.icmp "slt" %133, %131 : i64
      llvm.cond_br %134, ^bb10, ^bb11
    ^bb10:  // pred: ^bb9
      %135 = llvm.extractvalue %41[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
      %136 = llvm.mlir.constant(2 : index) : i64
      %137 = llvm.mul %128, %136  : i64
      %138 = llvm.add %137, %133  : i64
      %139 = llvm.getelementptr %135[%138] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
      %140 = llvm.load %139 : !llvm.ptr<f64>
      %141 = llvm.fmul %140, %140  : f64
      %142 = llvm.extractvalue %23[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
      %143 = llvm.mlir.constant(2 : index) : i64
      %144 = llvm.mul %128, %143  : i64
      %145 = llvm.add %144, %133  : i64
      %146 = llvm.getelementptr %142[%145] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
      llvm.store %141, %146 : !llvm.ptr<f64>
      %147 = llvm.add %133, %132  : i64
      llvm.br ^bb9(%147 : i64)
    ^bb11:  // pred: ^bb9
      %148 = llvm.add %128, %127  : i64
      llvm.br ^bb7(%148 : i64)
    ^bb12:  // pred: ^bb7
      %149 = llvm.mlir.addressof @frmt_spec : !llvm.ptr<array<4 x i8>>
      %150 = llvm.mlir.constant(0 : index) : i64
      %151 = llvm.getelementptr %149[%150, %150] : (!llvm.ptr<array<4 x i8>>, i64, i64) -> !llvm.ptr<i8>
      %152 = llvm.mlir.addressof @nl : !llvm.ptr<array<2 x i8>>
      %153 = llvm.mlir.constant(0 : index) : i64
      %154 = llvm.getelementptr %152[%153, %153] : (!llvm.ptr<array<2 x i8>>, i64, i64) -> !llvm.ptr<i8>
      %155 = llvm.mlir.constant(0 : index) : i64
      %156 = llvm.mlir.constant(3 : index) : i64
      %157 = llvm.mlir.constant(1 : index) : i64
      llvm.br ^bb13(%155 : i64)
    ^bb13(%158: i64):  // 2 preds: ^bb12, ^bb17
      %159 = llvm.icmp "slt" %158, %156 : i64
      llvm.cond_br %159, ^bb14, ^bb18
    ^bb14:  // pred: ^bb13
      %160 = llvm.mlir.constant(0 : index) : i64
      %161 = llvm.mlir.constant(2 : index) : i64
      %162 = llvm.mlir.constant(1 : index) : i64
      llvm.br ^bb15(%160 : i64)
    ^bb15(%163: i64):  // 2 preds: ^bb14, ^bb16
      %164 = llvm.icmp "slt" %163, %161 : i64
      llvm.cond_br %164, ^bb16, ^bb17
    ^bb16:  // pred: ^bb15
      %165 = llvm.extractvalue %23[1] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
      %166 = llvm.mlir.constant(2 : index) : i64
      %167 = llvm.mul %158, %166  : i64
      %168 = llvm.add %167, %163  : i64
      %169 = llvm.getelementptr %165[%168] : (!llvm.ptr<f64>, i64) -> !llvm.ptr<f64>
      %170 = llvm.load %169 : !llvm.ptr<f64>
      %171 = llvm.call @printf(%151, %170) : (!llvm.ptr<i8>, f64) -> i32
      %172 = llvm.add %163, %162  : i64
      llvm.br ^bb15(%172 : i64)
    ^bb17:  // pred: ^bb15
      %173 = llvm.call @printf(%154) : (!llvm.ptr<i8>) -> i32
      %174 = llvm.add %158, %157  : i64
      llvm.br ^bb13(%174 : i64)
    ^bb18:  // pred: ^bb13
      %175 = llvm.extractvalue %59[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
      %176 = llvm.bitcast %175 : !llvm.ptr<f64> to !llvm.ptr<i8>
      llvm.call @free(%176) : (!llvm.ptr<i8>) -> ()
      %177 = llvm.extractvalue %41[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
      %178 = llvm.bitcast %177 : !llvm.ptr<f64> to !llvm.ptr<i8>
      llvm.call @free(%178) : (!llvm.ptr<i8>) -> ()
      %179 = llvm.extractvalue %23[0] : !llvm.struct<(ptr<f64>, ptr<f64>, i64, array<2 x i64>, array<2 x i64>)> 
      %180 = llvm.bitcast %179 : !llvm.ptr<f64> to !llvm.ptr<i8>
      llvm.call @free(%180) : (!llvm.ptr<i8>) -> ()
      llvm.return
    }
    llvm.func @_mlir_ciface_main() attributes {llvm.emit_c_interface} {
      llvm.call @main() : () -> ()
      llvm.return
    }
  }
