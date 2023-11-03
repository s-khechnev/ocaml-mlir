open Ctypes

module Bindings (S : Cstubs.Types.TYPE) = struct
  open S

  (* Support.h Types *)
  module TypeID = struct
    type t

    let t : t structure typ = structure "MlirTypeID"
    let ptr = field t "ptr" (ptr void)
  end

  module TypeIDAllocator = struct
    type t

    let t : t structure typ = structure "MlirTypeIDAllocator"
    let ptr = field t "ptr" (ptr void)
  end

  module StringRef = struct
    type t

    let t : t structure typ = structure "MlirStringRef"
    let data = field t "data" string
    let length = field t "length" size_t
    let () = seal t
  end

  module LogicalResult = struct
    type t

    let t : t structure typ = structure "MlirLogicalResult"
    let value = field t "value" int8_t
    let () = seal t
  end

  (* IR.h Types *)
  module Context = struct
    type t

    let t : t structure typ = structure "MlirContext"
    let ptr = field t "ptr" (ptr void)
    let () = seal t
  end

  module Dialect = struct
    type t

    let t : t structure typ = structure "MlirDialect"
    let ptr = field t "ptr" (ptr void)
    let () = seal t
  end

  module DialectRegistry = struct
    type t

    let t : t structure typ = structure "MlirDialectRegistry"
    let ptr = field t "ptr" (ptr void)
    let () = seal t
  end

  module Operation = struct
    type t

    let t : t structure typ = structure "MlirOperation"
    let ptr = field t "ptr" (ptr void)
    let () = seal t
  end

  module OpOperand = struct
    type t

    let t : t structure typ = structure "MlirOpOperand"
    let ptr = field t "ptr" (ptr void)
    let () = seal t
  end

  module OpPrintingFlags = struct
    type t

    let t : t structure typ = structure "MlirOpPrintingFlags"
    let ptr = field t "ptr" (ptr void)
    let () = seal t
  end

  module Block = struct
    type t

    let t : t structure typ = structure "MlirBlock"
    let ptr = field t "ptr" (ptr void)
    let () = seal t
  end

  module Region = struct
    type t

    let t : t structure typ = structure "MlirRegion"
    let ptr = field t "ptr" (ptr void)
    let () = seal t
  end

  module SymbolTable = struct
    type t

    let t : t structure typ = structure "MlirSymbolTable"
    let ptr = field t "ptr" (ptr void)
    let () = seal t
  end

  module Attribute = struct
    type t

    let t : t structure typ = structure "MlirAttribute"
    let ptr = field t "ptr" (ptr void)
    let () = seal t
  end

  module Identifier = struct
    type t

    let t : t structure typ = structure "MlirIdentifier"
    let ptr = field t "ptr" (ptr void)
    let () = seal t
  end

  module Location = struct
    type t

    let t : t structure typ = structure "MlirLocation"
    let location_ptr = field t "ptr" (ptr void)
    let () = seal t
  end

  module Module = struct
    type t

    let t : t structure typ = structure "MlirModule"
    let ptr = field t "ptr" (ptr void)
    let () = seal t
  end

  module Type = struct
    type t

    let t : t structure typ = structure "MlirType"
    let ptr = field t "ptr" (ptr void)
    let () = seal t
  end

  module Value = struct
    type t

    let t : t structure typ = structure "MlirValue"
    let ptr = field t "ptr" (ptr void)
    let () = seal t
  end

  module NamedAttribute = struct
    type t

    let t : t structure typ = structure "MlirNamedAttribute"
    let name = field t "name" Identifier.t
    let attribute = field t "attribute" Attribute.t
    let () = seal t
  end

  module DialectHandle = struct
    type t

    let t : t structure typ = structure "MlirDialectHandle"
    let ptr = field t "ptr" (ptr void)
    let () = seal t
  end

  module OperationState = struct
    type t

    let t : t structure typ = structure "MlirOperationState"
    let name = field t "name" StringRef.t
    let location = field t "location" Location.t
    let nResults = field t "nResults" intptr_t
    let results = field t "results" (ptr Type.t)
    let nOperands = field t "nOperands" intptr_t
    let operands = field t "operands" (ptr Value.t)
    let nRegions = field t "nRegions" intptr_t
    let regions = field t "regions" (ptr Region.t)
    let nSuccessors = field t "nSuccessors" intptr_t
    let successors = field t "successors" (ptr Block.t)
    let nAttributes = field t "nAttributes" intptr_t
    let attributes = field t "attributes" (ptr Attribute.t)
    let enableResultTypeInference = field t "enableResultTypeInference" bool
    let () = seal t
  end

  (* Pass.h Types *)
  module Pass = struct
    type t

    let t : t structure typ = structure "MlirPass"
    let ptr = field t "ptr" (ptr void)
    let () = seal t
  end

  module ExternalPass = struct
    type t

    let t : t structure typ = structure "MlirExternalPass"
    let ptr = field t "ptr" (ptr void)
    let () = seal t
  end

  module PassManager = struct
    type t

    let t : t structure typ = structure "MlirPassManager"
    let ptr = field t "ptr" (ptr void)
    let () = seal t
  end

  module OpPassManager = struct
    type t

    let t : t structure typ = structure "MlirOpPassManager"
    let ptr = field t "ptr" (ptr void)
    let () = seal t
  end

  (* AffineExpr.h Types *)
  module AffineExpr = struct
    type t

    let t : t structure typ = structure "MlirAffineExpr"
    let ptr = field t "ptr" (ptr void)
    let () = seal t
  end

  (* AffineMap.h Types *)
  module AffineMap = struct
    type t

    let t : t structure typ = structure "MlirAffineMap"
    let ptr = field t "ptr" (ptr void)
    let () = seal t
  end

  (* Diagnostics.h Types *)
  module Diagnostic = struct
    type t

    type severity =
      | Error
      | Warning
      | Note
      | Remark

    let t : t structure typ = structure "MlirDiagnostic"
    let ptr = field t "ptr" (ptr void)
    let () = seal t
    let error = S.constant "MlirDiagnosticError" S.int64_t
    let warning = S.constant "MlirDiagnosticWarning" S.int64_t
    let note = S.constant "MlirDiagnosticNote" S.int64_t
    let remark = S.constant "MlirDiagnosticRemark" S.int64_t

    let severity =
      S.enum
        "MlirDiagnosticSeverity"
        ~typedef:true
        [ Error, error; Warning, warning; Note, note; Remark, remark ]
        ~unexpected:(fun _ -> failwith "unexpected Diagnostic enum")
  end
end
