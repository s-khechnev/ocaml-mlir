open Ctypes

module Bindings (F : FOREIGN) = struct
  open F

  module PassManager = struct
    (* Create a new top-level PassManager. *)
    let create =
      foreign "mlirPassManagerCreate" (Typs.Context.t @-> returning Typs.PassManager.t)


    (* Create a new top-level PassManager anchored on `anchorOp`. *)
    let create_on_operaion =
      foreign
        "mlirPassManagerCreateOnOperation"
        (Typs.Context.t @-> Typs.StringRef.t @-> returning Typs.PassManager.t)


    (* Destroy the provided PassManager. *)
    let destroy = foreign "mlirPassManagerDestroy" (Typs.PassManager.t @-> returning void)

    (* Checks if a PassManager is null. *)
    let is_null = foreign "mlirPassManagerIsNull" (Typs.PassManager.t @-> returning bool)

    (* Cast a top-level PassManager to a generic OpPassManager. *)
    let to_op_pass_manager =
      foreign
        "mlirPassManagerGetAsOpPassManager"
        (Typs.PassManager.t @-> returning Typs.OpPassManager.t)


    (* Run the provided `passManager` on the given `module`. *)
    let run =
      foreign
        "mlirPassManagerRun"
        (Typs.PassManager.t @-> Typs.Module.t @-> returning Typs.LogicalResult.t)


    (* Enable mlir-print-ir-after-all. *)
    let enable_ir_printing =
      foreign "mlirPassManagerEnableIRPrinting" (Typs.PassManager.t @-> returning void)


    (* Enable / disable verify-each. *)
    let enable_verifier =
      foreign
        "mlirPassManagerEnableVerifier"
        (Typs.PassManager.t @-> bool @-> returning void)


    (* Nest an OpPassManager under the top-level PassManager, the nested
     * passmanager will only run on operations matching the provided name.
     * The returned OpPassManager will be destroyed when the parent is destroyed.
     * To further nest more OpPassManager under the newly returned one, see
     * `mlirOpPassManagerNest` below. *)
    let nested_under =
      foreign
        "mlirPassManagerGetNestedUnder"
        (Typs.PassManager.t @-> Typs.StringRef.t @-> returning Typs.OpPassManager.t)


    (* Add a pass and transfer ownership to the provided top-level mlirPassManager.
     * If the pass is not a generic operation pass or a ModulePass, a new
     * OpPassManager is implicitly nested under the provided PassManager. *)
    let add_owned_pass =
      foreign
        "mlirPassManagerAddOwnedPass"
        (Typs.PassManager.t @-> Typs.Pass.t @-> returning void)
  end

  module OpPassManager = struct
    (* Nest an OpPassManager under the provided OpPassManager, the nested
     * passmanager will only run on operations matching the provided name.
     * The returned OpPassManager will be destroyed when the parent is destroyed. *)
    let nested_under =
      foreign
        "mlirOpPassManagerGetNestedUnder"
        (Typs.OpPassManager.t @-> Typs.StringRef.t @-> returning Typs.OpPassManager.t)


    (* Add a pass and transfer ownership to the provided mlirOpPassManager. If the
     * pass is not a generic operation pass or matching the type of the provided
     * PassManager, a new OpPassManager is implicitly nested under the provided
     * PassManager. *)
    let add_owned_pass =
      foreign
        "mlirOpPassManagerAddOwnedPass"
        (Typs.OpPassManager.t @-> Typs.Pass.t @-> returning void)


    let add_pipeline =
      foreign
        "mlirOpPassManagerAddPipeline"
        (Typs.OpPassManager.t
         @-> Typs.StringRef.t
         @-> Typs.string_callback
         @-> ptr void
         @-> returning void)


    (* Print a textual MLIR pass pipeline by sending chunks of the string
       * representation and forwarding `userData to `callback`. Note that the callback
       * may be called several times with consecutive chunks of the string. *)
    let print_pass_pipeline =
      foreign
        "mlirPrintPassPipeline"
        (Typs.OpPassManager.t @-> Typs.string_callback @-> ptr void @-> returning void)


    (* Parse a textual MLIR pass pipeline and add it to the provided OpPassManager. *)
    let parse_pass_pipeline =
      foreign
        "mlirParsePassPipeline"
        (Typs.OpPassManager.t
         @-> Typs.StringRef.t
         @-> Typs.string_callback
         @-> ptr void
         @-> returning Typs.LogicalResult.t)
  end

  module ExternalPass = struct
    module ExternalPassCallbacks = struct
      type t

      let t : t structure typ = structure "MlirExternalPassCallbacks"

      let construct =
        field t "construct" Ctypes.(Foreign.funptr (ptr void @-> returning void))


      let destruct =
        field t "destruct" Ctypes.(Foreign.funptr (ptr void @-> returning void))


      let initialize =
        field
          t
          "initialize"
          Ctypes.(
            Foreign.funptr (Typs.Context.t @-> ptr void @-> returning Typs.LogicalResult.t))


      let clone =
        field t "clone" Ctypes.(Foreign.funptr (ptr void @-> returning (ptr void)))


      let run =
        field
          t
          "run"
          Ctypes.(
            Foreign.funptr
              (Typs.Operation.t @-> Typs.ExternalPass.t @-> ptr void @-> returning void))


      let () = seal t
    end

    (* Creates an external `MlirPass` that calls the supplied `callbacks` using the
       supplied `userData`. If `opName` is empty, the pass is a generic operation
       pass. Otherwise it is an operation pass specific to the specified pass name. *)
    let create =
      foreign
        "mlirCreateExternalPass"
        (Typs.TypeID.t
         @-> Typs.StringRef.t
         @-> Typs.StringRef.t
         @-> Typs.StringRef.t
         @-> Typs.StringRef.t
         @-> intptr_t
         @-> ptr Typs.DialectHandle.t
         @-> ExternalPassCallbacks.t
         @-> ptr void
         @-> returning Typs.Pass.t)


    (* This signals that the pass has failed. This is only valid to call during
       the `run` callback of `MlirExternalPassCallbacks`.
       See Pass::signalPassFailure(). *)
    let signal_failure =
      foreign "mlirExternalPassSignalFailure" (Typs.ExternalPass.t @-> returning void)
  end
end
