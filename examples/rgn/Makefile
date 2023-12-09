includedir = $(shell llvm-config-16 --includedir)
cxxflags = $(shell llvm-config-16 --cxxflags)
td = dialect/include/rgn/Ops.td

tblgen:
	mkdir -p build/include/rgn
	mlir-tblgen-16 --gen-dialect-decls ${td} -I${includedir} -o build/include/rgn/Dialect.h.inc
	mlir-tblgen-16 --gen-dialect-defs ${td} -I${includedir} -o build/include/rgn/Dialect.cpp.inc
	mlir-tblgen-16 --gen-op-decls ${td} -I${includedir} -o build/include/rgn/Ops.h.inc
	mlir-tblgen-16 --gen-op-defs ${td} -I${includedir} -o build/include/rgn/Ops.cpp.inc

mlir_rgn: tblgen
	c++ -fPIC -Wall -c dialect/mlir/Dialect.cpp -Idialect/include -Ibuild/include/ ${cxxflags} -o build/RgnDialect.o
	c++ -shared build/RgnDialect.o ${cxxflags} -o build/libMLIRRgn.so

capi_rgn: mlir_rgn
	c++ -fPIC -Wall -c dialect/CAPI/Dialects.cpp -Idialect/include ${cxxflags} -Ibuild/include/ -o build/RgnCAPI.o
	c++ -shared build/RgnCAPI.o ${cxxflags} -o build/libRgnCAPI.so

clean:
	rm -rf build
