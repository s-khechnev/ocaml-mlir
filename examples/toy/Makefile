.PHONY: tblgen mlir_toy capi_toy clean

CXX ?= c++
INCLUDE_DIR = $(shell llvm-config-16 --includedir)
CXX_FLAGS = $(shell llvm-config-16 --cxxflags)
TD = dialect/include/toy/Ops.td

tblgen:
	mkdir -p build/include/toy
	mlir-tblgen-16 --gen-dialect-decls ${TD} -I${INCLUDE_DIR} -o build/include/toy/Dialect.h.inc
	mlir-tblgen-16 --gen-dialect-defs ${TD} -I${INCLUDE_DIR} -o build/include/toy/Dialect.cpp.inc
	mlir-tblgen-16 --gen-op-decls ${TD} -I${INCLUDE_DIR} -o build/include/toy/Ops.h.inc
	mlir-tblgen-16 --gen-op-defs ${TD} -I${INCLUDE_DIR} -o build/include/toy/Ops.cpp.inc
	mlir-tblgen-16 --gen-rewriters dialect/mlir/ToyCombine.td -Idialect/include -I${INCLUDE_DIR} -o build/ToyCombine.inc	

mlir_toy: tblgen
	${CXX} -fPIC -Wall -c dialect/mlir/Dialect.cpp -Idialect/include -Ibuild/include/ ${CXX_FLAGS} -o build/ToyDialect.o
	${CXX} -fPIC -Wall -c dialect/mlir/ToyCombine.cpp -Idialect/include -Ibuild -Ibuild/include/ ${CXX_FLAGS} -o build/ToyCombine.o
	${CXX} -fPIC -Wall -c dialect/mlir/LowerPrintToLLVM.cpp ${CXX_FLAGS} -Ibuild/include -Idialect/include -o build/LowerPrintToLLVM.o
	${CXX} -shared build/ToyDialect.o build/ToyCombine.o build/LowerPrintToLLVM.o ${CXX_FLAGS} -o build/libMLIRToy.so

capi_toy: mlir_toy
	${CXX} -fPIC -Wall -c dialect/CAPI/Dialects.cpp -Idialect/include -Ibuild/include/ ${CXX_FLAGS} -o build/ToyCAPI.o
	${CXX} -shared build/ToyCAPI.o ${CXX_FLAGS} -o build/libToyCAPI.so

clean:
	rm -rf build
