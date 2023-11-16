open Ctypes

module Bindings (F : FOREIGN) = struct
  open F

  (* Returns `true` if the given operation implements an interface identified by
     its TypeID. *)
  let implement_interface =
    foreign
      "mlirOperationImplementsInterface"
      (Typs.Operation.t @-> Typs.TypeID.t @-> returning bool)


  (* Returns `true` if the operation identified by its canonical string name
     implements the interface identified by its TypeID in the given context.
     Note that interfaces may be attached to operations in some contexts and not
     others. *)
  let implement_interface_static =
    foreign
      "mlirOperationImplementsInterfaceStatic"
      (Typs.StringRef.t @-> Typs.Context.t @-> Typs.TypeID.t @-> returning bool)


  module InferTypeOpInterface = struct
    (* Returns the interface TypeID of the InferTypeOpInterface. *)
    let type_id =
      foreign "mlirInferTypeOpInterfaceTypeID" (void @-> returning Typs.TypeID.t)


    (* These callbacks are used to return multiple types from functions while
       transferring ownership to the caller. The first argument is the number of
       consecutive elements pointed to by the second argument. The third argument
       is an opaque pointer forwarded to the callback by the caller. *)
    let types_callback =
      Ctypes.(
        Foreign.funptr (intptr_t @-> ptr Typs.Type.t @-> ptr void @-> returning void))


    (* Infers the return types of the operation identified by its canonical given
       the arguments that will be supplied to its generic builder. Calls `callback`
       with the types of inferred arguments, potentially several times, on success.
       Returns failure otherwise. *)
    let infer_return_types =
      foreign
        "mlirInferTypeOpInterfaceInferReturnTypes"
        (Typs.StringRef.t
         @-> Typs.Context.t
         @-> Typs.Location.t
         @-> intptr_t
         @-> ptr Typs.Value.t
         @-> Typs.Attribute.t
         @-> intptr_t
         @-> ptr Typs.Region.t
         @-> types_callback
         @-> ptr void
         @-> returning Typs.LogicalResult.t)
  end
end
