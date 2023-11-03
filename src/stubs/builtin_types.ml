open Ctypes

module Bindings (F : FOREIGN) = struct
  open F

  (*===----------------------------------------------------------------------===
   * Integer types.
   *===----------------------------------------------------------------------===*)

  module Integer = struct
    (* Checks whether the given type is an integer type. *)
    let is_integer = foreign "mlirTypeIsAInteger" (Typs.Type.t @-> returning bool)

    (* Creates a signless integer type of the given bitwidth in the context. The
     * type is owned by the context. *)
    let get =
      foreign "mlirIntegerTypeGet" (Typs.Context.t @-> uint @-> returning Typs.Type.t)


    (* Creates a signed integer type of the given bitwidth in the context. The type
     * is owned by the context. *)
    let signed =
      foreign
        "mlirIntegerTypeSignedGet"
        (Typs.Context.t @-> uint @-> returning Typs.Type.t)


    (* Creates an unsigned integer type of the given bitwidth in the context. The
     * type is owned by the context. *)
    let unsigned =
      foreign
        "mlirIntegerTypeUnsignedGet"
        (Typs.Context.t @-> uint @-> returning Typs.Type.t)


    (* Returns the bitwidth of an integer type. *)
    let width = foreign "mlirIntegerTypeGetWidth" (Typs.Type.t @-> returning uint)

    (* Checks whether the given integer type is signless. *)
    let is_signless = foreign "mlirIntegerTypeIsSignless" (Typs.Type.t @-> returning bool)

    (* Checks whether the given integer type is signed. *)
    let is_signed = foreign "mlirIntegerTypeIsSigned" (Typs.Type.t @-> returning bool)

    (* Checks whether the given integer type is unsigned. *)
    let is_unsigned = foreign "mlirIntegerTypeIsUnsigned" (Typs.Type.t @-> returning bool)
  end

  (*===----------------------------------------------------------------------===
   * Index type.
   *===----------------------------------------------------------------------===*)

  module Index = struct
    (* Checks whether the given type is an index type. *)
    let is_index = foreign "mlirTypeIsAIndex" (Typs.Type.t @-> returning bool)

    (* Creates an index type in the given context. The type is owned by the
     * context. *)
    let get = foreign "mlirIndexTypeGet" (Typs.Context.t @-> returning Typs.Type.t)
  end

  (*===----------------------------------------------------------------------===
   * Floating-point types.
   *===----------------------------------------------------------------------===*)

  module Float = struct
    (* Checks whether the given type is an f8E5M2 type. *)
    let is_f8e5m2 = foreign "mlirTypeIsAFloat8E5M2" (Typs.Type.t @-> returning bool)

    (* Creates an f8E5M2 type in the given context. The type is owned by the
     *  context *)
    let f8e5m2 = foreign "mlirFloat8E5M2TypeGet" (Typs.Context.t @-> returning Typs.Type.t)

    (* Checks whether the given type is an f8E4M3FN type. *)
    let is_f8E4M3FN = foreign "mlirTypeIsAFloat8E4M3FN" (Typs.Type.t @-> returning bool)

    (* Creates an f8E4M3FN type in the given context. The type is owned by the
     * context *)
    let f8e5m2 =
      foreign "mlirFloat8E4M3FNTypeGet" (Typs.Context.t @-> returning Typs.Type.t)


    (* Checks whether the given type is a bf16 type. *)
    let is_bf16 = foreign "mlirTypeIsABF16" (Typs.Type.t @-> returning bool)

    (* Creates a bf16 type in the given context. The type is owned by the
     * context. *)
    let bf16 = foreign "mlirBF16TypeGet" (Typs.Context.t @-> returning Typs.Type.t)

    (* Checks whether the given type is an f16 type. *)
    let is_f16 = foreign "mlirTypeIsAF16" (Typs.Type.t @-> returning bool)

    (* Creates an f16 type in the given context. The type is owned by the
     * context. *)
    let f16 = foreign "mlirF16TypeGet" (Typs.Context.t @-> returning Typs.Type.t)

    (* Checks whether the given type is an f32 type. *)
    let is_f32 = foreign "mlirTypeIsAF32" (Typs.Type.t @-> returning bool)

    (* Creates an f32 type in the given context. The type is owned by the
     * context. *)
    let f32 = foreign "mlirF32TypeGet" (Typs.Context.t @-> returning Typs.Type.t)

    (* Checks whether the given type is an f64 type. *)
    let is_f64 = foreign "mlirTypeIsAF64" (Typs.Type.t @-> returning bool)

    (* Creates a f64 type in the given context. The type is owned by the
     * context. *)
    let f64 = foreign "mlirF64TypeGet" (Typs.Context.t @-> returning Typs.Type.t)
  end

  (*===----------------------------------------------------------------------===
   * None type.
   *===----------------------------------------------------------------------===*)
  module None = struct
    (* Checks whether the given type is a None type. *)
    let is_none = foreign "mlirTypeIsANone" (Typs.Type.t @-> returning bool)

    (* Creates a None type in the given context. The type is owned by the
     * context. *)
    let get = foreign "mlirNoneTypeGet" (Typs.Context.t @-> returning Typs.Type.t)
  end

  (*===----------------------------------------------------------------------===
   * Complex type.
   *===----------------------------------------------------------------------===*)
  module Complex = struct
    (* Checks whether the given type is a Complex type. *)
    let is_complex = foreign "mlirTypeIsAComplex" (Typs.Type.t @-> returning bool)

    (* Creates a complex type with the given element type in the same context as
     * the element type. The type is owned by the context. *)
    let get = foreign "mlirComplexTypeGet" (Typs.Type.t @-> returning Typs.Type.t)

    (* Returns the element type of the given complex type. *)
    let element_type =
      foreign "mlirComplexTypeGetElementType" (Typs.Type.t @-> returning Typs.Type.t)
  end

  (*===----------------------------------------------------------------------===
   * Shaped type.
   *===----------------------------------------------------------------------===*)
  module Shaped = struct
    (* Checks whether the given type is a Shaped type. *)
    let is_shaped = foreign "mlirTypeIsAShaped" (Typs.Type.t @-> returning bool)

    (* Returns the element type of the shaped type. *)
    let get =
      foreign "mlirShapedTypeGetElementType" (Typs.Type.t @-> returning Typs.Type.t)


    (* Checks whether the given shaped type is ranked. *)
    let has_rank = foreign "mlirShapedTypeHasRank" (Typs.Type.t @-> returning bool)

    (* Returns the rank of the given ranked shaped type. *)
    let rank = foreign "mlirShapedTypeGetRank" (Typs.Type.t @-> returning int64_t)

    (* Checks whether the given shaped type has a static shape. *)
    let has_static_shape =
      foreign "mlirShapedTypeHasStaticShape" (Typs.Type.t @-> returning bool)


    (* Checks wither the dim-th dimension of the given shaped type is dynamic. *)
    let is_dynamic_dim =
      foreign "mlirShapedTypeIsDynamicDim" (Typs.Type.t @-> intptr_t @-> returning bool)


    (* Returns the dim-th dimension of the given ranked shaped type. *)
    let dim_size =
      foreign "mlirShapedTypeGetDimSize" (Typs.Type.t @-> intptr_t @-> returning int64_t)


    (* Checks whether the given value is used as a placeholder for dynamic sizes
     * in shaped types. *)
    let is_dynamic_size =
      foreign "mlirShapedTypeIsDynamicSize" (int64_t @-> returning bool)


    (* Checks whether the given value is used as a placeholder for dynamic strides
     * and offsets in shaped types. *)
    let is_dynamic_stride_or_offset =
      foreign "mlirShapedTypeIsDynamicStrideOrOffset" (int64_t @-> returning bool)


    (* Returns the value indicating a dynamic stride or offset in a shaped type.
       Prefer mlirShapedTypeGetDynamicStrideOrOffset to direct comparisons with
       this value. *)
    let dynamic_stride_or_offset =
      foreign "mlirShapedTypeGetDynamicStrideOrOffset" (void @-> returning int64_t)
  end

  (*===----------------------------------------------------------------------===
   * Vector type.
   *===----------------------------------------------------------------------===*)
  module Vector = struct
    (* Checks whether the given type is a Vector type. *)
    let is_vector = foreign "mlirTypeIsAVector" (Typs.Type.t @-> returning bool)

    (* Creates a vector type of the shape identified by its rank and dimensions,
     * with the given element type in the same context as the element type. The type
     * is owned by the context. *)
    let get =
      foreign
        "mlirVectorTypeGet"
        (intptr_t @-> ptr int64_t @-> Typs.Type.t @-> returning Typs.Type.t)


    (* Same as "mlirVectorTypeGet" but returns a nullptr wrapping MlirType on
     * illegal arguments, emitting appropriate diagnostics. *)
    let get_checked =
      foreign
        "mlirVectorTypeGetChecked"
        (Typs.Location.t
         @-> intptr_t
         @-> ptr int64_t
         @-> Typs.Type.t
         @-> returning Typs.Type.t)
  end

  (*===----------------------------------------------------------------------===
   * Ranked / Unranked Tensor type.
   *===----------------------------------------------------------------------===*)
  module Tensor = struct
    (* Checks whether the given type is a Tensor type. *)
    let is_tensor = foreign "mlirTypeIsATensor" (Typs.Type.t @-> returning bool)

    (* Checks whether the given type is a ranked tensor type. *)
    let is_ranked_tensor =
      foreign "mlirTypeIsARankedTensor" (Typs.Type.t @-> returning bool)


    (* Checks whether the given type is an unranked tensor type. *)
    let is_unranked_tensor =
      foreign "mlirTypeIsAUnrankedTensor" (Typs.Type.t @-> returning bool)


    (* Creates a tensor type of a fixed rank with the given shape, element type,
       and optional encoding in the same context as the element type. The type is
       owned by the context. Tensor types without any specific encoding field
       should assign mlirAttributeGetNull() to this parameter. *)
    let ranked =
      foreign
        "mlirRankedTensorTypeGet"
        (intptr_t
         @-> ptr int64_t
         @-> Typs.Type.t
         @-> Typs.Attribute.t
         @-> returning Typs.Type.t)


    (* Same as "mlirRankedTensorTypeGet" but returns a nullptr wrapping MlirType on
     * illegal arguments, emitting appropriate diagnostics. *)
    let ranked_checked =
      foreign
        "mlirRankedTensorTypeGetChecked"
        (Typs.Location.t
         @-> intptr_t
         @-> ptr int64_t
         @-> Typs.Type.t
         @-> Typs.Attribute.t
         @-> returning Typs.Type.t)


    (* Gets the 'encoding' attribute from the ranked tensor type, returning a
     * null attribute if none. *)
    let ranked_encoding =
      foreign
        "mlirRankedTensorTypeGetEncoding"
        (Typs.Type.t @-> returning Typs.Attribute.t)


    (* Creates an unranked tensor type with the given element type in the same
     * context as the element type. The type is owned by the context. *)
    let unranked =
      foreign "mlirUnrankedTensorTypeGet" (Typs.Type.t @-> returning Typs.Type.t)


    (* Same as "mlirUnrankedTensorTypeGet" but returns a nullptr wrapping MlirType
     * on illegal arguments, emitting appropriate diagnostics. *)
    let unranked_checked =
      foreign
        "mlirUnrankedTensorTypeGetChecked"
        (Typs.Location.t @-> Typs.Type.t @-> returning Typs.Type.t)
  end

  (*===----------------------------------------------------------------------===
   * Ranked / Unranked MemRef type.
   *===----------------------------------------------------------------------===*)

  module MemRef = struct
    (* Checks whether the given type is a MemRef type. *)
    let is_memref = foreign "mlirTypeIsAMemRef" (Typs.Type.t @-> returning bool)

    (* Checks whether the given type is an UnrankedMemRef type. *)
    let is_unranked_memref =
      foreign "mlirTypeIsAUnrankedMemRef" (Typs.Type.t @-> returning bool)


    (* Creates a MemRef type with the given rank and shape, a potentially empty
     * list of affine layout maps, the given memory space and element type, in the
     * same context as element type. The type is owned by the context. *)
    let get =
      foreign
        "mlirMemRefTypeGet"
        (Typs.Type.t
         @-> intptr_t
         @-> ptr int64_t
         @-> Typs.Attribute.t
         @-> Typs.Attribute.t
         @-> returning Typs.Type.t)


    (* Same as "mlirMemRefTypeGet" but returns a nullptr-wrapping MlirType o
       illegal arguments, emitting appropriate diagnostics. *)
    let checked =
      foreign
        "mlirMemRefTypeGetChecked"
        (Typs.Location.t
         @-> Typs.Type.t
         @-> intptr_t
         @-> ptr int64_t
         @-> Typs.Attribute.t
         @-> Typs.Attribute.t
         @-> returning Typs.Type.t)


    (* Creates a MemRef type with the given rank, shape, memory space and element
     * type in the same context as the element type. The type has no affine maps,
     * i.e. represents a default row-major contiguous memref. The type is owned by
     * the context. *)
    let contiguous =
      foreign
        "mlirMemRefTypeContiguousGet"
        (Typs.Type.t
         @-> intptr_t
         @-> ptr int64_t
         @-> Typs.Attribute.t
         @-> returning Typs.Type.t)


    (* Same as "mlirMemRefTypeContiguousGet" but returns a nullptr wrapping
     * MlirType on illegal arguments, emitting appropriate diagnostics. *)
    let contiguous_checked =
      foreign
        "mlirMemRefTypeContiguousGetChecked"
        (Typs.Location.t
         @-> Typs.Type.t
         @-> intptr_t
         @-> ptr int64_t
         @-> Typs.Attribute.t
         @-> returning Typs.Type.t)


    (* Creates an Unranked MemRef type with the given element type and in the given
     * memory space. The type is owned by the context of element type. *)
    let unranked =
      foreign
        "mlirUnrankedMemRefTypeGet"
        (Typs.Type.t @-> Typs.Attribute.t @-> returning Typs.Type.t)


    (* Same as "mlirUnrankedMemRefTypeGet" but returns a nullptr wrapping
     * MlirType on illegal arguments, emitting appropriate diagnostics. *)
    let unranked_checked =
      foreign
        "mlirUnrankedMemRefTypeGetChecked"
        (Typs.Location.t @-> Typs.Type.t @-> Typs.Attribute.t @-> returning Typs.Type.t)


    (* Returns the layout of the given MemRef type. *)
    let layout =
      foreign "mlirMemRefTypeGetLayout" (Typs.Type.t @-> returning Typs.Attribute.t)


    (* Returns the affine map of the given MemRef type. *)
    let affine_map =
      foreign "mlirMemRefTypeGetAffineMap" (Typs.Type.t @-> returning Typs.AffineMap.t)


    (* Returns the memory space of the given MemRef type. *)
    let memory_space =
      foreign "mlirMemRefTypeGetMemorySpace" (Typs.Type.t @-> returning Typs.Attribute.t)


    (* Returns the memory spcae of the given Unranked MemRef type. *)
    let unranked_memory_space =
      foreign
        "mlirUnrankedMemrefGetMemorySpace"
        (Typs.Type.t @-> returning Typs.Attribute.t)
  end

  (*===----------------------------------------------------------------------===
   * Tuple type.
   *===----------------------------------------------------------------------===*)
  module Tuple = struct
    (* Checks whether the given type is a tuple type. *)
    let is_tuple = foreign "mlirTypeIsATuple" (Typs.Type.t @-> returning bool)

    (* Creates a tuple type that consists of the given list of elemental types. The
     * type is owned by the context. *)
    let get =
      foreign
        "mlirTupleTypeGet"
        (Typs.Context.t @-> intptr_t @-> ptr Typs.Type.t @-> returning Typs.Type.t)


    (* Returns the number of types contained in a tuple. *)
    let num_types = foreign "mlirTupleTypeGetNumTypes" (Typs.Type.t @-> returning intptr_t)

    (* Returns the pos-th type in the tuple type. *)
    let get_type =
      foreign "mlirTupleTypeGetType" (Typs.Type.t @-> intptr_t @-> returning Typs.Type.t)
  end

  (*===----------------------------------------------------------------------===
   * Function type.
   *===----------------------------------------------------------------------===*)

  module Function = struct
    (* Checks whether the given type is a function type. *)
    let is_function = foreign "mlirTypeIsAFunction" (Typs.Type.t @-> returning bool)

    (* Creates a function type, mapping a list of input types to result types. *)
    let get =
      foreign
        "mlirFunctionTypeGet"
        (Typs.Context.t
         @-> intptr_t
         @-> ptr Typs.Type.t
         @-> intptr_t
         @-> ptr Typs.Type.t
         @-> returning Typs.Type.t)


    (* Returns the number of input types. *)
    let num_inputs =
      foreign "mlirFunctionTypeGetNumInputs" (Typs.Type.t @-> returning intptr_t)


    (* Returns the number of result types. *)
    let num_results =
      foreign "mlirFunctionTypeGetNumResults" (Typs.Type.t @-> returning intptr_t)


    (* Returns the pos-th input type. *)
    let input =
      foreign
        "mlirFunctionTypeGetInput"
        (Typs.Type.t @-> intptr_t @-> returning Typs.Type.t)


    (* Returns the pos-th result type. *)
    let result =
      foreign
        "mlirFunctionTypeGetResult"
        (Typs.Type.t @-> intptr_t @-> returning Typs.Type.t)
  end

  module Opaque = struct
    (* Checks whether the given type is an opaque type. *)
    let is_opaque = foreign "mlirTypeIsAOpaque" (Typs.Type.t @-> returning bool)

    (* Creates an opaque type in the given context associated with the dialect
       identified by its namespace. The type contains opaque byte data of the
       specified length (data need not be null-terminated). *)
    let get =
      foreign
        "mlirOpaqueTypeGet"
        (Typs.Context.t
         @-> Typs.StringRef.t
         @-> Typs.StringRef.t
         @-> returning Typs.Type.t)


    (* Returns the namespace of the dialect with which the given opaque type
       is associated. The namespace string is owned by the context. *)
    let dialect_namespace =
      foreign
        "mlirOpaqueTypeGetDialectNamespace"
        (Typs.Type.t @-> returning Typs.StringRef.t)


    (* Returns the raw data as a string reference. The data remains live as long as
       the context in which the type lives. *)
    let data = foreign "mlirOpaqueTypeGetData" (Typs.Type.t @-> returning Typs.StringRef.t)
  end
end
