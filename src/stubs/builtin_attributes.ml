open Ctypes

module Bindings (F : FOREIGN) = struct
  open F

  (* Returns an empty attribute. *)
  let null = foreign "mlirAttributeGetNull" (void @-> returning Typs.Attribute.t)

  (*===----------------------------------------------------------------------===
   * Affine map attribute.
   *===----------------------------------------------------------------------===*)

  module AffineMap = struct
    (* Checks whether the given attribute is an affine map attribute. *)
    let is_affine_map =
      foreign "mlirAttributeIsAAffineMap" (Typs.Attribute.t @-> returning bool)


    (* Creates an affine map attribute wrapping the given map. The attribute
     * belongs to the same context as the affine map. *)
    let get =
      foreign "mlirAffineMapAttrGet" (Typs.AffineMap.t @-> returning Typs.Attribute.t)


    (* Returns the affine map wrapped in the given affine map attribute. *)
    let value =
      foreign "mlirAffineMapAttrGetValue" (Typs.Attribute.t @-> returning Typs.AffineMap.t)
  end

  (*===----------------------------------------------------------------------===
   * Array attribute.
   *===----------------------------------------------------------------------===*)

  module Array = struct
    (* Checks whether the given attribute is an array attribute. *)
    let is_array = foreign "mlirAttributeIsAArray" (Typs.Attribute.t @-> returning bool)

    (* Creates an array element containing the given list of elements in the given
     * context. *)
    let get =
      foreign
        "mlirArrayAttrGet"
        (Typs.Context.t
         @-> intptr_t
         @-> ptr Typs.Attribute.t
         @-> returning Typs.Attribute.t)


    (* Returns the number of elements stored in the given array attribute. *)
    let num_elements =
      foreign "mlirArrayAttrGetNumElements" (Typs.Attribute.t @-> returning intptr_t)


    (* Returns pos-th element stored in the given array attribute. *)
    let element =
      foreign
        "mlirArrayAttrGetElement"
        (Typs.Attribute.t @-> intptr_t @-> returning Typs.Attribute.t)
  end

  (*===----------------------------------------------------------------------===
   * Dictionary attribute.
   *===----------------------------------------------------------------------===*)

  module Dictionary = struct
    (* Checks whether the given attribute is a dictionary attribute. *)
    let is_dicationary =
      foreign "mlirAttributeIsADictionary" (Typs.Attribute.t @-> returning bool)


    (* Creates a dictionary attribute containing the given list of elements in the
     * provided context. *)
    let get =
      foreign
        "mlirDictionaryAttrGet"
        (Typs.Context.t
         @-> intptr_t
         @-> ptr Typs.NamedAttribute.t
         @-> returning Typs.Attribute.t)


    (* Returns the number of attributes contained in a dictionary attribute. *)
    let num_elements =
      foreign "mlirDictionaryAttrGetNumElements" (Typs.Attribute.t @-> returning intptr_t)


    (* Returns pos-th element of the given dictionary attribute. *)
    let element =
      foreign
        "mlirDictionaryAttrGetElement"
        (Typs.Attribute.t @-> intptr_t @-> returning Typs.NamedAttribute.t)


    (* Returns the dictionary attribute element with the given name or NULL if the
     * given name does not exist in the dictionary. *)
    let element_by_name =
      foreign
        "mlirDictionaryAttrGetElementByName"
        (Typs.Attribute.t @-> Typs.StringRef.t @-> returning Typs.Attribute.t)
  end

  (*===----------------------------------------------------------------------===
   * Floating point attribute.
   *===----------------------------------------------------------------------===*)

  module Float = struct
    (* TODO: add support for APFloat and APInt to LLVM IR C API, then expose the
     * relevant functions here. *)

    (* Checks whether the given attribute is a floating point attribute. *)
    let is_float = foreign "mlirAttributeIsAFloat" (Typs.Attribute.t @-> returning bool)

    (* Creates a floating point attribute in the given context with the given
     * double value and double-precision FP semantics. *)
    let get =
      foreign
        "mlirFloatAttrDoubleGet"
        (Typs.Context.t @-> Typs.Type.t @-> double @-> returning Typs.Attribute.t)


    (* Same as "mlirFloatAttrDoubleGet", but if the type is not valid for a
     * construction of a FloatAttr, returns a null MlirAttribute. *)
    let get_checked =
      foreign
        "mlirFloatAttrDoubleGetChecked"
        (Typs.Location.t @-> Typs.Type.t @-> double @-> returning Typs.Attribute.t)


    (* Returns the value stored in the given floating point attribute, interpreting
     * the value as double. *)
    let value =
      foreign "mlirFloatAttrGetValueDouble" (Typs.Attribute.t @-> returning double)
  end

  (*===----------------------------------------------------------------------===
   * Integer attribute.
   *===----------------------------------------------------------------------===*)

  module Integer = struct
    (* TODO: add support for APFloat and APInt to LLVM IR C API, then expose the
     * relevant functions here. *)

    (* Checks whether the given attribute is an integer attribute. *)
    let is_integer =
      foreign "mlirAttributeIsAInteger" (Typs.Attribute.t @-> returning bool)


    (* Creates an integer attribute of the given type with the given integer
     * value. *)
    let get =
      foreign "mlirIntegerAttrGet" (Typs.Type.t @-> int64_t @-> returning Typs.Attribute.t)


    (* Returns the value stored in the given integer attribute, assuming the value
     * fits into a 64-bit integer. *)
    let value_int =
      foreign "mlirIntegerAttrGetValueInt" (Typs.Attribute.t @-> returning int64_t)


    (* Returns the value stored in the given integer attribute, assuming the value
       is of signed type and fits into a signed 64-bit integer. *)
    let value_sint =
      foreign "mlirIntegerAttrGetValueSInt" (Typs.Attribute.t @-> returning int64_t)


    (* Returns the value stored in the given integer attribute, assuming the value
       is of unsigned type and fits into an unsigned 64-bit integer. *)
    let value_uint =
      foreign "mlirIntegerAttrGetValueUInt" (Typs.Attribute.t @-> returning uint64_t)
  end

  (*===----------------------------------------------------------------------===
   * Bool attribute.
   *===----------------------------------------------------------------------===*)

  module Bool = struct
    (* Checks whether the given attribute is a bool attribute. *)
    let is_bool = foreign "mlirAttributeIsABool" (Typs.Attribute.t @-> returning bool)

    (* Creates a bool attribute in the given context with the given value. *)
    let get =
      foreign "mlirBoolAttrGet" (Typs.Context.t @-> int @-> returning Typs.Attribute.t)


    (* Returns the value stored in the given bool attribute. *)
    let value = foreign "mlirBoolAttrGetValue" (Typs.Attribute.t @-> returning bool)
  end

  (*===----------------------------------------------------------------------===
   * Integer set attribute.
   *===----------------------------------------------------------------------===*)

  module IntegerSet = struct
    (* Checks whether the given attribute is an integer set attribute. *)
    let is_integer_set =
      foreign "mlirAttributeIsAIntegerSet" (Typs.Attribute.t @-> returning bool)
  end

  (*===----------------------------------------------------------------------===
   * Opaque attribute.
   *===----------------------------------------------------------------------===*)

  module Opaque = struct
    (* Checks whether the given attribute is an opaque attribute. *)
    let is_opaque = foreign "mlirAttributeIsAOpaque" (Typs.Attribute.t @-> returning bool)

    (* Creates an opaque attribute in the given context associated with the dialect
     * identified by its namespace. The attribute contains opaque byte data of the
     * specified length (data need not be null-terminated). *)
    let get =
      foreign
        "mlirOpaqueAttrGet"
        (Typs.Context.t
         @-> Typs.StringRef.t
         @-> intptr_t
         @-> ptr char
         @-> Typs.Type.t
         @-> returning Typs.Attribute.t)


    (* Returns the namespace of the dialect with which the given opaque attribute
     * is associated. The namespace string is owned by the context. *)
    let namespace =
      foreign
        "mlirOpaqueAttrGetDialectNamespace"
        (Typs.Attribute.t @-> returning Typs.StringRef.t)


    (* Returns the raw data as a string reference. The data remains live as long as
     * the context in which the attribute lives. *)
    let data =
      foreign "mlirOpaqueAttrGetData" (Typs.Attribute.t @-> returning Typs.StringRef.t)
  end

  (*===----------------------------------------------------------------------===
   * String attribute.
   *===----------------------------------------------------------------------===*)

  module String = struct
    (* checks whether the given attribute is a string attribute. *)
    let is_string = foreign "mlirAttributeIsAString" (Typs.Attribute.t @-> returning bool)

    (* Creates a string attribute in the given context containing the given string. *)
    let get =
      foreign
        "mlirStringAttrGet"
        (Typs.Context.t @-> Typs.StringRef.t @-> returning Typs.Attribute.t)


    (* Creates a string attribute in the given context containing the given string.
     * Additionally, the attribute has the given type. *)

    let typed_get =
      foreign
        "mlirStringAttrTypedGet"
        (Typs.Type.t @-> Typs.StringRef.t @-> returning Typs.Attribute.t)


    (* Returns the attribute values as a string reference. The data remains live as
     * long as the context in which the attribute lives. *)
    let value =
      foreign "mlirStringAttrGetValue" (Typs.Attribute.t @-> returning Typs.StringRef.t)
  end

  (*===----------------------------------------------------------------------===
   * SymbolRef attribute.
   *===----------------------------------------------------------------------===*)

  module SymbolRef = struct
    (* Checks whether the given attribute is a symbol reference attribute. *)
    let is_symbol_ref =
      foreign "mlirAttributeIsASymbolRef" (Typs.Attribute.t @-> returning bool)


    (* Creates a symbol reference attribute in the given context referencing a
     * symbol identified by the given string inside a list of nested references.
     * Each of the references in the list must not be nested. *)
    let get =
      foreign
        "mlirSymbolRefAttrGet"
        (Typs.Context.t
         @-> Typs.StringRef.t
         @-> intptr_t
         @-> ptr Typs.Attribute.t
         @-> returning Typs.Attribute.t)


    (* Returns the string reference to the root referenced symbol. The data remains
     * live as long as the context in which the attribute lives. *)
    let root_ref =
      foreign
        "mlirSymbolRefAttrGetRootReference"
        (Typs.Attribute.t @-> returning Typs.StringRef.t)


    (* Returns the string reference to the leaf referenced symbol. The data remains
     * live as long as the context in which the attribute lives. *)
    let leaf_ref =
      foreign
        "mlirSymbolRefAttrGetLeafReference"
        (Typs.Attribute.t @-> returning Typs.StringRef.t)


    (* Returns the number of references nested in the given symbol reference
     * attribute. *)
    let num_nested_refs =
      foreign
        "mlirSymbolRefAttrGetNumNestedReferences"
        (Typs.Attribute.t @-> returning intptr_t)


    (* Returns pos-th reference nested in the given symbol reference attribute. *)
    let nested_ref =
      foreign
        "mlirSymbolRefAttrGetNestedReference"
        (Typs.Attribute.t @-> intptr_t @-> returning Typs.Attribute.t)
  end

  (*===----------------------------------------------------------------------===
   * Flat SymbolRef attribute.
   *===----------------------------------------------------------------------===*)

  module FlatSymbolRef = struct
    (* Checks whether the given attribute is a flat symbol reference attribute. *)
    let is_flat_symbol_ref =
      foreign "mlirAttributeIsAFlatSymbolRef" (Typs.Attribute.t @-> returning bool)


    (* Creates a flat symbol reference attribute in the given context referencing a
     * symbol identified by the given string. *)
    let get =
      foreign
        "mlirFlatSymbolRefAttrGet"
        (Typs.Context.t @-> Typs.StringRef.t @-> returning Typs.Attribute.t)


    (* Returns the referenced symbol as a string reference. The data remains live
     * as long as the context in which the attribute lives. *)
    let value =
      foreign
        "mlirFlatSymbolRefAttrGetValue"
        (Typs.Attribute.t @-> returning Typs.StringRef.t)
  end

  (*===----------------------------------------------------------------------===
   * Type attribute.
   *===----------------------------------------------------------------------===*)

  module Type = struct
    (* Checks whether the given attribute is a type attribute. *)
    let is_type = foreign "mlirAttributeIsAType" (Typs.Attribute.t @-> returning bool)

    (* Creates a type attribute wrapping the given type in the same context as the
     * type. *)
    let get = foreign "mlirTypeAttrGet" (Typs.Type.t @-> returning Typs.Attribute.t)

    (* Returns the type stored in the given type attribute. *)
    let value = foreign "mlirTypeAttrGetValue" (Typs.Attribute.t @-> returning Typs.Type.t)
  end

  (*===----------------------------------------------------------------------===
   * Unit attribute.
   *===----------------------------------------------------------------------===*)

  module Unit = struct
    (* Checks whether the given attribute is a unit attribute. *)
    let is_unit = foreign "mlirAttributeIsAUnit" (Typs.Attribute.t @-> returning bool)

    (* Creates a unit attribute in the given context. *)
    let get = foreign "mlirUnitAttrGet" (Typs.Context.t @-> returning Typs.Attribute.t)
  end

  (*===----------------------------------------------------------------------===
   * Elements attributes.
   *===----------------------------------------------------------------------===*)

  module Elements = struct
    (* Checks whether the given attribute is an elements attribute. *)
    let is_elements =
      foreign "mlirAttributeIsAElements" (Typs.Attribute.t @-> returning bool)


    (* Returns the element at the given rank-dimensional index. *)
    let get =
      foreign
        "mlirElementsAttrGetValue"
        (Typs.Attribute.t @-> intptr_t @-> ptr uint64_t @-> returning Typs.Attribute.t)


    (* Checks whether the given rank-dimensional index is valid in the given
     * elements attribute. *)
    let is_valid_index =
      foreign
        "mlirElementsAttrIsValidIndex"
        (Typs.Attribute.t @-> intptr_t @-> ptr uint64_t @-> returning bool)


    (* Gets the total number of elements in the given elements attribute. In order
     * to iterate over the attribute, obtain its type, which must be a statically
     * shaped type and use its sizes to build a multi-dimensional index. *)
    let num_elements =
      foreign "mlirElementsAttrGetNumElements" (Typs.Attribute.t @-> returning int64_t)
  end

  (* ===----------------------------------------------------------------------===
   *  Dense array attribute.
   * ===----------------------------------------------------------------------===*)
  module Dense = struct
    module Array = struct
      (* Checks whether the given attribute is a dense array attribute. *)
      let is_bool =
        foreign "mlirAttributeIsADenseBoolArray" (Typs.Attribute.t @-> returning bool)


      let is_i8 =
        foreign "mlirAttributeIsADenseI8Array" (Typs.Attribute.t @-> returning bool)


      let is_i16 =
        foreign "mlirAttributeIsADenseI16Array" (Typs.Attribute.t @-> returning bool)


      let is_i32 =
        foreign "mlirAttributeIsADenseI32Array" (Typs.Attribute.t @-> returning bool)


      let is_i64 =
        foreign "mlirAttributeIsADenseI64Array" (Typs.Attribute.t @-> returning bool)


      let is_f32 =
        foreign "mlirAttributeIsADenseF32Array" (Typs.Attribute.t @-> returning bool)


      let is_f64 =
        foreign "mlirAttributeIsADenseF64Array" (Typs.Attribute.t @-> returning bool)


      (* Create a dense array attribute with the given elements. *)
      let bool =
        foreign
          "mlirDenseBoolArrayGet"
          (Typs.Context.t @-> intptr_t @-> ptr int @-> returning Typs.Attribute.t)


      let i8 =
        foreign
          "mlirDenseI8ArrayGet"
          (Typs.Context.t @-> intptr_t @-> ptr int8_t @-> returning Typs.Attribute.t)


      let i16 =
        foreign
          "mlirDenseI16ArrayGet"
          (Typs.Context.t @-> intptr_t @-> ptr int16_t @-> returning Typs.Attribute.t)


      let i32 =
        foreign
          "mlirDenseI32ArrayGet"
          (Typs.Context.t @-> intptr_t @-> ptr int32_t @-> returning Typs.Attribute.t)


      let i64 =
        foreign
          "mlirDenseI64ArrayGet"
          (Typs.Context.t @-> intptr_t @-> ptr int64_t @-> returning Typs.Attribute.t)


      let f32 =
        foreign
          "mlirDenseF32ArrayGet"
          (Typs.Context.t @-> intptr_t @-> ptr float @-> returning Typs.Attribute.t)


      let f64 =
        foreign
          "mlirDenseF64ArrayGet"
          (Typs.Context.t @-> intptr_t @-> ptr double @-> returning Typs.Attribute.t)


      (* Get the size of a dense array *)
      let num_elements =
        foreign "mlirDenseArrayGetNumElements" (Typs.Attribute.t @-> returning intptr_t)


      (*  Get an element of a dense array. *)
      let bool_elt =
        foreign
          "mlirDenseBoolArrayGetElement"
          (Typs.Attribute.t @-> intptr_t @-> returning Ctypes.(bool))


      let i8_elt =
        foreign
          "mlirDenseI8ArrayGetElement"
          (Typs.Attribute.t @-> intptr_t @-> returning int8_t)


      let i16_elt =
        foreign
          "mlirDenseI8ArrayGetElement"
          (Typs.Attribute.t @-> intptr_t @-> returning int16_t)


      let i32_elt =
        foreign
          "mlirDenseI32ArrayGetElement"
          (Typs.Attribute.t @-> intptr_t @-> returning int32_t)


      let i64_elt =
        foreign
          "mlirDenseI64ArrayGetElement"
          (Typs.Attribute.t @-> intptr_t @-> returning int64_t)


      let f32_elt =
        foreign
          "mlirDenseF32ArrayGetElement"
          (Typs.Attribute.t @-> intptr_t @-> returning float)


      let f64_elt =
        foreign
          "mlirDenseF64ArrayGetElement"
          (Typs.Attribute.t @-> intptr_t @-> returning double)
    end

    (*===----------------------------------------------------------------------===
     * Dense elements attribute.
     *===----------------------------------------------------------------------===*)

    module Elements = struct
      (* TODO: decide on the interface and add support for complex elements. *)
      (* TODO: add support for APFloat and APInt to LLVM IR C API, then expose the
       * relevant functions here. *)

      (* Checks whether the given attribute is a dense elements attribute. *)
      let is_dense =
        foreign "mlirAttributeIsADenseElements" (Typs.Attribute.t @-> returning bool)


      let is_dense_int =
        foreign "mlirAttributeIsADenseIntElements" (Typs.Attribute.t @-> returning bool)


      let is_dense_fpe =
        foreign "mlirAttributeIsADenseFPElements" (Typs.Attribute.t @-> returning bool)


      (* Creates a dense elements attribute with the given Shaped type and elements
       * in the same context as the type. *)
      let get =
        foreign
          "mlirDenseElementsAttrGet"
          (Typs.Type.t
           @-> intptr_t
           @-> ptr Typs.Attribute.t
           @-> returning Typs.Attribute.t)


      (* Creates a dense elements attribute with the given Shaped type and elements
         populated from a packed, row-major opaque buffer of contents.

         The format of the raw buffer is a densely packed array of values that
         can be bitcast to the storage format of the element type specified.
         Types that are not byte aligned will be:
         - For bitwidth > 1: Rounded up to the next byte.
         - For bitwidth = 1: Packed into 8bit bytes with bits corresponding to
           the linear order of the shape type from MSB to LSB, padded to on the
           right.

         A raw buffer of a single element (or for 1-bit, a byte of value 0 or 255)
         will be interpreted as a splat. User code should be prepared for additional,
         conformant patterns to be identified as splats in the future. *)
      let raw_buffer =
        foreign
          "mlirDenseElementsAttrRawBufferGet"
          (Typs.Type.t @-> size_t @-> ptr void @-> returning Typs.Attribute.t)


      (* Creates a dense elements attribute with the given Shaped type containing a
       * single replicated element (splat). *)
      let splat_get =
        foreign
          "mlirDenseElementsAttrSplatGet"
          (Typs.Type.t @-> Typs.Attribute.t @-> returning Typs.Attribute.t)


      let bool_splat_get =
        foreign
          "mlirDenseElementsAttrBoolSplatGet"
          (Typs.Type.t @-> bool @-> returning Typs.Attribute.t)


      let uint8_splat_get =
        foreign
          "mlirDenseElementsAttrUInt8SplatGet"
          (Typs.Type.t @-> uint8_t @-> returning Typs.Attribute.t)


      let int8_splat_get =
        foreign
          "mlirDenseElementsAttrInt8SplatGet"
          (Typs.Type.t @-> int8_t @-> returning Typs.Attribute.t)


      let uint32_splat_get =
        foreign
          "mlirDenseElementsAttrUInt32SplatGet"
          (Typs.Type.t @-> uint32_t @-> returning Typs.Attribute.t)


      let int32_splat_get =
        foreign
          "mlirDenseElementsAttrInt32SplatGet"
          (Typs.Type.t @-> int32_t @-> returning Typs.Attribute.t)


      let uint64_splat_get =
        foreign
          "mlirDenseElementsAttrUInt64SplatGet"
          (Typs.Type.t @-> uint64_t @-> returning Typs.Attribute.t)


      let int64_splat_get =
        foreign
          "mlirDenseElementsAttrInt64SplatGet"
          (Typs.Type.t @-> int64_t @-> returning Typs.Attribute.t)


      let float_splat_get =
        foreign
          "mlirDenseElementsAttrFloatSplatGet"
          (Typs.Type.t @-> float @-> returning Typs.Attribute.t)


      let double_splat_get =
        foreign
          "mlirDenseElementsAttrDoubleSplatGet"
          (Typs.Type.t @-> double @-> returning Typs.Attribute.t)


      (* Creates a dense elements attribute with the given shaped type from elements
       * of a specific type. Expects the element type of the shaped type to match the
       * data element type. *)
      let bool_get =
        foreign
          "mlirDenseElementsAttrBoolGet"
          (Typs.Type.t @-> intptr_t @-> ptr int @-> returning Typs.Attribute.t)


      let uint8_get =
        foreign
          "mlirDenseElementsAttrUInt8Get"
          (Typs.Type.t @-> intptr_t @-> ptr uint8_t @-> returning Typs.Attribute.t)


      let int8_get =
        foreign
          "mlirDenseElementsAttrInt8Get"
          (Typs.Type.t @-> intptr_t @-> ptr int8_t @-> returning Typs.Attribute.t)


      let uint16_get =
        foreign
          "mlirDenseElementsAttrUInt16Get"
          (Typs.Type.t @-> intptr_t @-> ptr uint16_t @-> returning Typs.Attribute.t)


      let int16_get =
        foreign
          "mlirDenseElementsAttrInt16Get"
          (Typs.Type.t @-> intptr_t @-> ptr int16_t @-> returning Typs.Attribute.t)


      let uint32_get =
        foreign
          "mlirDenseElementsAttrUInt32Get"
          (Typs.Type.t @-> intptr_t @-> ptr uint32_t @-> returning Typs.Attribute.t)


      let int32_get =
        foreign
          "mlirDenseElementsAttrInt32Get"
          (Typs.Type.t @-> intptr_t @-> ptr int32_t @-> returning Typs.Attribute.t)


      let uint64_get =
        foreign
          "mlirDenseElementsAttrUInt64Get"
          (Typs.Type.t @-> intptr_t @-> ptr uint64_t @-> returning Typs.Attribute.t)


      let int64_get =
        foreign
          "mlirDenseElementsAttrInt64Get"
          (Typs.Type.t @-> intptr_t @-> ptr int64_t @-> returning Typs.Attribute.t)


      let float_get =
        foreign
          "mlirDenseElementsAttrFloatGet"
          (Typs.Type.t @-> intptr_t @-> ptr float @-> returning Typs.Attribute.t)


      let double_get =
        foreign
          "mlirDenseElementsAttrDoubleGet"
          (Typs.Type.t @-> intptr_t @-> ptr double @-> returning Typs.Attribute.t)


      let b_float16_get =
        foreign
          "mlirDenseElementsAttrBFloat16Get"
          (Typs.Type.t @-> intptr_t @-> ptr uint16_t @-> returning Typs.Attribute.t)


      let float16_get =
        foreign
          "mlirDenseElementsAttrBFloat16Get"
          (Typs.Type.t @-> intptr_t @-> ptr uint16_t @-> returning Typs.Attribute.t)


      (* Creates a dense elements attribute with the given shaped type from string
       * elements. *)
      let string_get =
        foreign
          "mlirDenseElementsAttrStringGet"
          (Typs.Type.t
           @-> intptr_t
           @-> ptr Typs.StringRef.t
           @-> returning Typs.Attribute.t)


      (* Creates a dense elements attribute that has the same data as the given dense
       * elements attribute and a different shaped type. The new type must have the
       * same total number of elements. *)
      let reshape_get =
        foreign
          "mlirDenseElementsAttrReshapeGet"
          (Typs.Attribute.t @-> Typs.Type.t @-> returning Typs.Attribute.t)


      (* Checks whether the given dense elements attribute contains a single
       * replicated value (splat). *)
      let is_splat =
        foreign "mlirDenseElementsAttrIsSplat" (Typs.Attribute.t @-> returning bool)


      (* Returns the single replicated value (splat) of a specific type contained by
       * the given dense elements attribute. *)
      let splat_value =
        foreign
          "mlirDenseElementsAttrGetSplatValue"
          (Typs.Attribute.t @-> returning Typs.Attribute.t)


      let bool_splat_value =
        foreign
          "mlirDenseElementsAttrGetBoolSplatValue"
          (Typs.Attribute.t @-> returning int)


      let int8_splat_value =
        foreign
          "mlirDenseElementsAttrGetInt8SplatValue"
          (Typs.Attribute.t @-> returning int8_t)


      let uint8_splat_value =
        foreign
          "mlirDenseElementsAttrGetUInt8SplatValue"
          (Typs.Attribute.t @-> returning uint8_t)


      let int32_splat_value =
        foreign
          "mlirDenseElementsAttrGetInt32SplatValue"
          (Typs.Attribute.t @-> returning int32_t)


      let uint32_splat_value =
        foreign
          "mlirDenseElementsAttrGetUInt32SplatValue"
          (Typs.Attribute.t @-> returning uint32_t)


      let int64_splat_value =
        foreign
          "mlirDenseElementsAttrGetInt64SplatValue"
          (Typs.Attribute.t @-> returning int64_t)


      let uint64_splat_value =
        foreign
          "mlirDenseElementsAttrGetUInt64SplatValue"
          (Typs.Attribute.t @-> returning uint64_t)


      let float_splat_value =
        foreign
          "mlirDenseElementsAttrGetFloatSplatValue"
          (Typs.Attribute.t @-> returning float)


      let double_splat_value =
        foreign
          "mlirDenseElementsAttrGetDoubleSplatValue"
          (Typs.Attribute.t @-> returning double)


      let string_splat_value =
        foreign
          "mlirDenseElementsAttrGetStringSplatValue"
          (Typs.Attribute.t @-> returning Typs.StringRef.t)


      (* Returns the pos-th value (flat contiguous indexing) of a specific type
       * contained by the given dense elements attribute. *)
      let bool_value =
        foreign
          "mlirDenseElementsAttrGetBoolValue"
          (Typs.Attribute.t @-> intptr_t @-> returning bool)


      let int8_value =
        foreign
          "mlirDenseElementsAttrGetInt8Value"
          (Typs.Attribute.t @-> intptr_t @-> returning int8_t)


      let uint8_value =
        foreign
          "mlirDenseElementsAttrGetUInt8Value"
          (Typs.Attribute.t @-> intptr_t @-> returning uint8_t)


      let int16_value =
        foreign
          "mlirDenseElementsAttrGetInt16Value"
          (Typs.Attribute.t @-> intptr_t @-> returning int16_t)


      let uint16_value =
        foreign
          "mlirDenseElementsAttrGetUInt16Value"
          (Typs.Attribute.t @-> intptr_t @-> returning uint16_t)


      let int32_value =
        foreign
          "mlirDenseElementsAttrGetInt32Value"
          (Typs.Attribute.t @-> intptr_t @-> returning int32_t)


      let uint32_value =
        foreign
          "mlirDenseElementsAttrGetUInt32Value"
          (Typs.Attribute.t @-> intptr_t @-> returning uint32_t)


      let int64_value =
        foreign
          "mlirDenseElementsAttrGetInt64Value"
          (Typs.Attribute.t @-> intptr_t @-> returning int64_t)


      let uint64_value =
        foreign
          "mlirDenseElementsAttrGetUInt64Value"
          (Typs.Attribute.t @-> intptr_t @-> returning uint64_t)


      let float_value =
        foreign
          "mlirDenseElementsAttrGetFloatValue"
          (Typs.Attribute.t @-> intptr_t @-> returning float)


      let double_value =
        foreign
          "mlirDenseElementsAttrGetDoubleValue"
          (Typs.Attribute.t @-> intptr_t @-> returning double)


      let string_value =
        foreign
          "mlirDenseElementsAttrGetStringValue"
          (Typs.Attribute.t @-> intptr_t @-> returning Typs.StringRef.t)


      (* Returns the raw data of the given dense elements attribute. *)
      let raw_data =
        foreign
          "mlirDenseElementsAttrGetRawData"
          (Typs.Attribute.t @-> returning (ptr void))
    end
  end

  (*===----------------------------------------------------------------------===
   * Resource blob attributes.
   *===----------------------------------------------------------------------===*)

  module Resource = struct
    let unmanaged_bool =
      foreign
        "mlirUnmanagedDenseBoolResourceElementsAttrGet"
        (Typs.Type.t
         @-> Typs.StringRef.t
         @-> intptr_t
         @-> ptr int
         @-> returning Typs.Attribute.t)


    let unmanaged_uint8 =
      foreign
        "mlirUnmanagedDenseUInt8ResourceElementsAttrGet"
        (Typs.Type.t
         @-> Typs.StringRef.t
         @-> intptr_t
         @-> ptr uint8_t
         @-> returning Typs.Attribute.t)


    let unmanaged_int8 =
      foreign
        "mlirUnmanagedDenseInt8ResourceElementsAttrGet"
        (Typs.Type.t
         @-> Typs.StringRef.t
         @-> intptr_t
         @-> ptr int8_t
         @-> returning Typs.Attribute.t)


    let unmanaged_uint16 =
      foreign
        "mlirUnmanagedDenseUInt16ResourceElementsAttrGet"
        (Typs.Type.t
         @-> Typs.StringRef.t
         @-> intptr_t
         @-> ptr uint16_t
         @-> returning Typs.Attribute.t)


    let unmanaged_int16 =
      foreign
        "mlirUnmanagedDenseInt16ResourceElementsAttrGet"
        (Typs.Type.t
         @-> Typs.StringRef.t
         @-> intptr_t
         @-> ptr int16_t
         @-> returning Typs.Attribute.t)


    let unmanaged_uint32 =
      foreign
        "mlirUnmanagedDenseUInt32ResourceElementsAttrGet"
        (Typs.Type.t
         @-> Typs.StringRef.t
         @-> intptr_t
         @-> ptr uint32_t
         @-> returning Typs.Attribute.t)


    let unmanaged_int32 =
      foreign
        "mlirUnmanagedDenseInt32ResourceElementsAttrGet"
        (Typs.Type.t
         @-> Typs.StringRef.t
         @-> intptr_t
         @-> ptr int32_t
         @-> returning Typs.Attribute.t)


    let unmanaged_uint64 =
      foreign
        "mlirUnmanagedDenseUInt64ResourceElementsAttrGet"
        (Typs.Type.t
         @-> Typs.StringRef.t
         @-> intptr_t
         @-> ptr uint64_t
         @-> returning Typs.Attribute.t)


    let unmanaged_int64 =
      foreign
        "mlirUnmanagedDenseInt64ResourceElementsAttrGet"
        (Typs.Type.t
         @-> Typs.StringRef.t
         @-> intptr_t
         @-> ptr int64_t
         @-> returning Typs.Attribute.t)


    let unmanaged_float =
      foreign
        "mlirUnmanagedDenseFloatResourceElementsAttrGet"
        (Typs.Type.t
         @-> Typs.StringRef.t
         @-> intptr_t
         @-> ptr float
         @-> returning Typs.Attribute.t)


    let unmanaged_double =
      foreign
        "mlirUnmanagedDenseDoubleResourceElementsAttrGet"
        (Typs.Type.t
         @-> Typs.StringRef.t
         @-> intptr_t
         @-> ptr double
         @-> returning Typs.Attribute.t)


    let bool_value =
      foreign
        "mlirDenseBoolResourceElementsAttrGetValue"
        (Typs.Attribute.t @-> intptr_t @-> returning bool)


    let int8_value =
      foreign
        "mlirDenseInt8ResourceElementsAttrGetValue"
        (Typs.Attribute.t @-> intptr_t @-> returning int8_t)


    let uint8_value =
      foreign
        "mlirDenseUInt8ResourceElementsAttrGetValue"
        (Typs.Attribute.t @-> intptr_t @-> returning uint8_t)


    let int16_value =
      foreign
        "mlirDenseInt16ResourceElementsAttrGetValue"
        (Typs.Attribute.t @-> intptr_t @-> returning int16_t)


    let uint16_value =
      foreign
        "mlirDenseUInt16ResourceElementsAttrGetValue"
        (Typs.Attribute.t @-> intptr_t @-> returning uint16_t)


    let int32_value =
      foreign
        "mlirDenseInt32ResourceElementsAttrGetValue"
        (Typs.Attribute.t @-> intptr_t @-> returning int32_t)


    let uint32_value =
      foreign
        "mlirDenseUInt32ResourceElementsAttrGetValue"
        (Typs.Attribute.t @-> intptr_t @-> returning uint32_t)


    let int64_value =
      foreign
        "mlirDenseInt64ResourceElementsAttrGetValue"
        (Typs.Attribute.t @-> intptr_t @-> returning int64_t)


    let uint64_value =
      foreign
        "mlirDenseUInt64ResourceElementsAttrGetValue"
        (Typs.Attribute.t @-> intptr_t @-> returning uint64_t)


    let float_value =
      foreign
        "mlirDenseFloatResourceElementsAttrGetValue"
        (Typs.Attribute.t @-> intptr_t @-> returning float)


    let double_value =
      foreign
        "mlirDenseDoubleResourceElementsAttrGetValue"
        (Typs.Attribute.t @-> intptr_t @-> returning double)
  end

  (*===----------------------------------------------------------------------===
   * Sparse elements attribute.
   *===----------------------------------------------------------------------===*)

  module Sparse = struct
    (* Checks whether the given attribute is a sparse elements attribute. *)
    let is_sparse =
      foreign "mlirAttributeIsASparseElements" (Typs.Attribute.t @-> returning bool)


    (* Creates a sparse elements attribute of the given shape from a list of
     * indices and a list of associated values. Both lists are expected to be dense
     * elements attributes with the same number of elements. The list of indices is
     * expected to contain 64-bit integers. The attribute is created in the same
     * context as the type. *)
    let create =
      foreign
        "mlirSparseElementsAttribute"
        (Typs.Type.t
         @-> Typs.Attribute.t
         @-> Typs.Attribute.t
         @-> returning Typs.Attribute.t)


    (* Returns the dense elements attribute containing 64-bit integer indices of
     * non-null elements in the given sparse elements attribute. *)
    let indices =
      foreign
        "mlirSparseElementsAttrGetIndices"
        (Typs.Attribute.t @-> returning Typs.Attribute.t)


    (* Returns the dense elements attribute containing the non-null elements in the
     * given sparse elements attribute. *)
    let values =
      foreign
        "mlirSparseElementsAttrGetValues"
        (Typs.Attribute.t @-> returning Typs.Attribute.t)
  end

  (*===----------------------------------------------------------------------===
   * Strided layout attribute.
   *===----------------------------------------------------------------------===*)

  module Strided = struct
    (* Checks wheather the given attribute is a strided layout attribute. *)
    let is_strided =
      foreign "mlirAttributeIsAStridedLayout" (Typs.Attribute.t @-> returning bool)


    (* Creates a strided layout attribute from given strides and offset. *)
    let create =
      foreign
        "mlirStridedLayoutAttrGet"
        (Typs.Context.t
         @-> int64_t
         @-> intptr_t
         @-> ptr int64_t
         @-> returning Typs.Attribute.t)


    (* Returns the offset in the given strided layout layout attribute. *)
    let offset =
      foreign "mlirStridedLayoutAttrGetOffset" (Typs.Attribute.t @-> returning int64_t)


    (* Returns the number of strides in the given strided layout attribute. *)
    let num_strides =
      foreign
        "mlirStridedLayoutAttrGetNumStrides"
        (Typs.Attribute.t @-> returning intptr_t)


    (* Returns the pos-th stride stored in the given strided layout attribute. *)
    let stride =
      foreign
        "mlirStridedLayoutAttrGetStride"
        (Typs.Attribute.t @-> intptr_t @-> returning int64_t)
  end
end
