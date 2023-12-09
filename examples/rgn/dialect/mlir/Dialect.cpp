#include "rgn/Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::rgn;

#include "rgn/Dialect.cpp.inc"

//===----------------------------------------------------------------------===//
// RgnDialect
//===----------------------------------------------------------------------===//

/// Dialect initialization, the instance will be owned by the context. This is
/// the point of registration of types and operations for the dialect.
void RgnDialect::initialize()
{
  addOperations<
#define GET_OP_LIST
#include "rgn/Ops.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "rgn/Ops.cpp.inc"
