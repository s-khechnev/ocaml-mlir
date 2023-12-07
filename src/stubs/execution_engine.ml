open Ctypes
open Typs

module Bindings (F : FOREIGN) = struct
  open F

  (** Creates an ExecutionEngine for the provided ModuleOp. The ModuleOp is
      expected to be "translatable" to LLVM IR (only contains operations in
      dialects that implement the `LLVMTranslationDialectInterface`). The module
      ownership stays with the client and can be destroyed as soon as the call
      returns. `optLevel` is the optimization level to be used for transformation
      and code generation. LLVM passes at `optLevel` are run before code
      generation. The number and array of paths corresponding to shared libraries
      that will be loaded are specified via `numPaths` and `sharedLibPaths`
      respectively. *)
  let create =
    foreign
      "mlirExecutionEngineCreate"
      (Module.t
       @-> int
       @-> int
       @-> ptr StringRef.t
       @-> bool
       @-> returning ExecutionEngine.t)


  (** Destroy an ExecutionEngine instance. *)
  let destroy = foreign "mlirExecutionEngineDestroy" (ExecutionEngine.t @-> returning void)

  (** Checks whether an execution engine is null. *)
  let is_null = foreign "mlirExecutionEngineIsNull" (ExecutionEngine.t @-> returning bool)

  (** Lookup the wrapper of the native function in the execution engine with the
      given name, returns nullptr if the function can't be looked-up. *)
  let invoke_packed =
    foreign
      "mlirExecutionEngineInvokePacked"
      (ExecutionEngine.t @-> StringRef.t @-> ptr void @-> returning LogicalResult.t)


  (** Lookup the wrapper of the native function in the execution engine with the
      given name, returns nullptr if the function can't be looked-up. *)
  let lookup_packed =
    foreign
      "mlirExecutionEngineLookupPacked"
      (ExecutionEngine.t @-> StringRef.t @-> returning (ptr void))


  (** Lookup a native function in the execution engine by name, returns nullptr
      if the name can't be looked-up. *)
  let lookup =
    foreign
      "mlirExecutionEngineLookup"
      (ExecutionEngine.t @-> StringRef.t @-> returning (ptr void))


  (** Register a symbol with the jit: this symbol will be accessible to the jitted
      code. *)
  let register_symbol =
    foreign
      "mlirExecutionEngineRegisterSymbol"
      (ExecutionEngine.t @-> StringRef.t @-> ptr void @-> returning void)


  (** Dump as an object in `fileName`. *)
  let dump_to_obj_file =
    foreign
      "mlirExecutionEngineDumpToObjectFile"
      (ExecutionEngine.t @-> StringRef.t @-> returning void)
end
