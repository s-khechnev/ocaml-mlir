#ifndef RGN_OPS
#define RGN_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Provide a definition of the 'rgn' dialect in the ODS framework so that we
// can define our operations.
def Rgn_Dialect : Dialect {
  let name = "rgn";
  let cppNamespace = "::mlir::rgn";
  let useFoldAPI = kEmitFoldAdaptorFolder;
}

// Base class for rgn dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Rgn_Op<string mnemonic, list<Trait> traits = []> :
    Op<Rgn_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Rgn Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ValOp
//===----------------------------------------------------------------------===//

def ValOp : Rgn_Op<"val", [Pure]> {
  let summary = "Region value operation";
  let description = [{
    Creates an SSA value which names a region definition.

    Example:

    ```mlir
    %x = rgn.val () ({
      %c = constant 42 : i32
      rgn.return %c : i32
    }) : () -> (i32)
    ```
  }];

  let results = (outs Variadic<AnyType>:$results);

  let regions = (region AnyRegion:$region);
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def ReturnOp : Rgn_Op<"return", [Pure, HasParent<"ValOp">, Terminator]> {
  let summary = "Function return operation";
  let description = [{
    The `rgn.return` operation represents a return operation within a rgn.val.
    The operation takes variable number of operands and produces no results.
    The operand number and types must match the signature of the rgn.val
    that contains the operation.

    Example:

    ```mlir
    %x = rgn.val () ({
      %c = constant 42 : i32
      rgn.return %c : i32
    }) : () -> (i32)
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

//===----------------------------------------------------------------------===//
// RunOp
//===----------------------------------------------------------------------===//

def RunOp : Rgn_Op<"run", [Pure, Terminator]> {
  let summary = "Run region operation";
  let description = [{
    The `rgn.run` is terminator operaion which transfers control
    flow to a region with the supplied arguments.
    
    Example:

    ```mlir
    %x = rgn.val () ({
      %c = constant 42 : i32
      rgn.return %c : i32
    }) : () -> (i32)

    rgn.run %x : i32
    ```
  }];

  let arguments = (ins AnyType:$operand);

  let assemblyFormat = "$operand `:` type($operand) attr-dict";
}

#endif // RGN_OPS
