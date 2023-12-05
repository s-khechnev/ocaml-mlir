includedir = $(shell llvm-config-16 --includedir)
cxxflags = $(shell llvm-config-16 --cxxflags)
td = dialect/include/toy/Ops.td

tblgen:
	mkdir -p build/include/toy
	mlir-tblgen-16 --gen-dialect-decls ${td} -I${includedir} -o build/include/toy/Dialect.h.inc
	mlir-tblgen-16 --gen-dialect-defs ${td} -I${includedir} -o build/include/toy/Dialect.cpp.inc
	mlir-tblgen-16 --gen-op-decls ${td} -I${includedir} -o build/include/toy/Ops.h.inc
	mlir-tblgen-16 --gen-op-defs ${td} -I${includedir} -o build/include/toy/Ops.cpp.inc
	mlir-tblgen-16 --gen-rewriters dialect/mlir/ToyCombine.td -Idialect/include -I${includedir} -o build/ToyCombine.inc

mlir_toy: tblgen
	c++ -fPIC -Wall -c dialect/mlir/Dialect.cpp -Idialect/include -I${includedir} -Ibuild/include/ ${cxxflags} -o build/ToyDialect.o
	c++ -fPIC -Wall -c dialect/mlir/ToyCombine.cpp -Idialect/include -I${includedir} -Ibuild -Ibuild/include/ ${cxxflags} -o build/ToyCombine.o
	c++ -shared build/ToyDialect.o build/ToyCombine.o ${cxxflags} -o build/libMLIRToy.so

capi_toy: mlir_toy
	c++ -fPIC -Wall -c dialect/CAPI/Dialects.cpp -Idialect/include -I${includedir} ${cxxflags} -Ibuild/include/ -o build/ToyCAPI.o
	c++ -shared build/ToyCAPI.o ${cxxflags} -o build/libToyCAPI.so

clean:
	rm -rf build
