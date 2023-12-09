#ifndef RGN_DIALECT_H
#define RGN_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

/// Include the auto-generated header file containing the declaration of the rgn
/// dialect.
#include "rgn/Dialect.h.inc"

/// Include the auto-generated header file containing the declarations of the
/// rgn operations.
#define GET_OP_CLASSES
#include "rgn/Ops.h.inc"

#endif // RGN_DIALECT_H
