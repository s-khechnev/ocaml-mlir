open Ctypes

(* Support API *)
module Bindings (F : FOREIGN) = struct
  open F

  (* StringRef API *)
  module StringRef = struct
    let create =
      foreign "mlirStringRefCreate" (string @-> size_t @-> returning Typs.StringRef.t)


    let of_string =
      foreign "mlirStringRefCreateFromCString" (string @-> returning Typs.StringRef.t)


    (* Returns true if two string references are equal, false otherwise. *)
    let equal =
      foreign
        "mlirStringRefEqual"
        (Typs.StringRef.t @-> Typs.StringRef.t @-> returning bool)


    let to_string d =
      let i = getf d Typs.StringRef.length |> Unsigned.Size_t.to_int in
      let s = getf d Typs.StringRef.data in
      String.sub s 0 i
  end

  (* LogicalResult API *)
  module LogicalResult = struct
    (* Checks if the given logical result represents a success. *)
    let is_success =
      foreign "mlirLogicalResultIsSuccess" (Typs.LogicalResult.t @-> returning bool)


    (* Checks if the given logical result represents a failure. *)
    let is_faiure =
      foreign "mlirLogicalResultIsFailure" (Typs.LogicalResult.t @-> returning bool)


    (* Creates a logical result representing a success. *)
    let success =
      foreign "mlirLogicalResultSuccess" (void @-> returning Typs.LogicalResult.t)


    (* Creates a logical result representing a failure. *)
    let failure =
      foreign "mlirLogicalResultFailure" (void @-> returning Typs.LogicalResult.t)
  end

  (*===----------------------------------------------------------------------===
     TypeID API.
    ===----------------------------------------------------------------------===*)

  module TypeID = struct
    (* ptr` must be 8 byte aligned and unique to a type valid for the duration of
       the returned type id's usage *)
    let create = foreign "mlirTypeIDCreate" (ptr void @-> returning Typs.TypeID.t)

    (* Checks whether a type id is null. *)
    let is_null = foreign "mlirTypeIDIsNull" (Typs.TypeID.t @-> returning bool)

    (* Checks if two type ids are equal. *)
    let equal =
      foreign "mlirTypeIDEqual" (Typs.TypeID.t @-> Typs.TypeID.t @-> returning bool)


    (* Returns the hash value of the type id. *)
    let hash_value = foreign "mlirTypeIDHashValue" (Typs.TypeID.t @-> returning size_t)
  end

  (*===----------------------------------------------------------------------===
     TypeIDAllocator API.
    ===----------------------------------------------------------------------===*)
  module TypeIDAllocator = struct
    (* Creates a type id allocator for dynamic type id creation *)
    let create =
      foreign "mlirTypeIDAllocatorCreate" (void @-> returning Typs.TypeIDAllocator.t)


    (* Deallocates the allocator and all allocated type ids *)
    let destroy =
      foreign "mlirTypeIDAllocatorDestroy" (Typs.TypeIDAllocator.t @-> returning void)


    (* Allocates a type id that is valid for the lifetime of the allocator *)
    let allocate_type_id =
      foreign
        "mlirTypeIDAllocatorAllocateTypeID"
        (Typs.TypeIDAllocator.t @-> returning Typs.TypeID.t)
  end
end
