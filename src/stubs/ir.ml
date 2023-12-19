open Ctypes

module Bindings (F : FOREIGN) = struct
  open F

  (*===----------------------------------------------------------------------===
   * Context API.
   *===----------------------------------------------------------------------===*)

  module Context = struct
    open Typs.Context

    (* Creates an MLIR context and transfers its ownership to the caller. *)
    let create = foreign "mlirContextCreate" (void @-> returning t)

    (* Checks if two contexts are equal. *)
    let equal = foreign "mlirContextEqual" (t @-> t @-> returning bool)

    (* Checks whether a context is null. *)
    let is_null = foreign "mlirContextIsNull" (t @-> returning bool)

    (* Takes an MLIR context owned by the caller and destroys it. *)
    let destroy = foreign "mlirContextDestroy" (t @-> returning void)

    (* Sets whether unregistered dialects are allowed in this context. *)
    let set_allow_unregistered_dialects =
      foreign "mlirContextSetAllowUnregisteredDialects" (t @-> bool @-> returning void)


    (* Returns whether the context allows unregistered dialects. *)
    let get_allow_unregistered_dialects =
      foreign "mlirContextGetAllowUnregisteredDialects" (t @-> returning bool)


    (* Returns the number of dialects registered with the given context. A
       registered dialect will be loaded if needed by the parser. *)
    let num_registered_dialects =
      foreign "mlirContextGetNumRegisteredDialects" (t @-> returning long)


    (* Append the contents of the given dialect registry to the registry associated
       with the context. *)
    let append_dialect_registry =
      foreign
        "mlirContextAppendDialectRegistry"
        (t @-> Typs.DialectRegistry.t @-> returning void)


    (* Returns the number of dialects loaded by the context. *)
    let num_loaded_dialects =
      foreign "mlirContextGetNumLoadedDialects" (t @-> returning long)


    (* Gets the dialect instance owned by the given context using the dialect
       namespace to identify it, loads (i.e., constructs the instance of) the
       dialect if necessary. If the dialect is not registered with the context,
       returns null. Use mlirContextLoad<Name>Dialect to load an unregistered
       dialect. *)
    let get_or_load_dialect =
      foreign
        "mlirContextGetOrLoadDialect"
        (t @-> Typs.StringRef.t @-> returning Typs.Dialect.t)


    (* Set threading mode (must be set to false to mlir-print-ir-after-all). *)
    let enable_multithreading =
      foreign "mlirContextEnableMultithreading" (t @-> bool @-> returning void)


    (* Eagerly loads all available dialects registered with a context, making
       them available for use for IR construction. *)
    let load_all_available_dialects =
      foreign "mlirContextLoadAllAvailableDialects" (t @-> returning void)


    (* Returns whether the given fully-qualified operation (i.e.
       'dialect.operation') is registered with the context. This will return true
       if the dialect is loaded and the operation is registered within the
       dialect. *)
    let is_registered_operation =
      foreign
        "mlirContextIsRegisteredOperation"
        (t @-> Typs.StringRef.t @-> returning bool)
  end

  (*===----------------------------------------------------------------------===
   * Dialect API.
   *===----------------------------------------------------------------------===*)

  module Dialect = struct
    open Typs.Dialect

    (* Returns the context that owns the dialect. *)
    let context = foreign "mlirDialectGetContext" (t @-> returning Typs.Context.t)

    (* Checks if the dialect is null. *)
    let is_null = foreign "mlirDialectIsNull" (t @-> returning bool)

    (* Checks if two dialects that belong to the same context are equal. Dialects
       from different contexts will not compare equal. *)
    let equal = foreign "mlirDialectEqual" (t @-> t @-> returning bool)

    (* Returns the namespace of the given dialect. *)
    let namespace = foreign "mlirDialectGetNamespace" (t @-> returning Typs.StringRef.t)
  end

  (* ===----------------------------------------------------------------------===
     DialectHandle API.
     Registration entry-points for each dialect are declared using the common
     MLIR_DECLARE_DIALECT_REGISTRATION_CAPI macro, which takes the dialect
     API name (i.e. "Func", "Tensor", "Linalg") and namespace (i.e. "func",
     "tensor", "linalg"). The following declarations are produced:
  
        * Gets the above hook methods in struct form for a dialect by namespace.
        * This is intended to facilitate dynamic lookup and registration of
        * dialects via a plugin facility based on shared library symbol lookup.
     const MlirDialectHandle *mlirGetDialectHandle__{NAMESPACE}__();
  
     This is done via a common macro to facilitate future expansion to
     registration schemes.
     ===----------------------------------------------------------------------=== *)
  module DialectHandle = struct
    (* Returns the namespace associated with the provided dialect handle. *)
    let namespace =
      foreign
        "mlirDialectHandleGetNamespace"
        (Typs.DialectHandle.t @-> returning Typs.StringRef.t)


    (* Inserts the dialect associated with the provided dialect handle into the
       provided dialect registry. *)
    let insert =
      foreign
        "mlirDialectHandleInsertDialect"
        (Typs.DialectHandle.t @-> Typs.DialectRegistry.t @-> returning void)


    (* Registers the dialect associated with the provided dialect handle. *)
    let register =
      foreign
        "mlirDialectHandleRegisterDialect"
        (Typs.DialectHandle.t @-> Typs.Context.t @-> returning void)


    (* Loads the dialect associated with the provided dialect handle. *)
    let load =
      foreign
        "mlirDialectHandleLoadDialect"
        (Typs.DialectHandle.t @-> Typs.Context.t @-> returning Typs.Dialect.t)
  end

  (* ===----------------------------------------------------------------------===
   *  DialectRegistry API.
   * ===----------------------------------------------------------------------===*)

  module DialectRegistry = struct
    (* Creates a dialect registry and transfers its ownership to the caller. *)
    let create =
      foreign "mlirDialectRegistryCreate" (void @-> returning Typs.DialectRegistry.t)


    (* Checks if the dialect registry is null. *)
    let is_null =
      foreign "mlirDialectRegistryIsNull" (Typs.DialectRegistry.t @-> returning bool)


    (* Takes a dialect registry owned by the caller and destroys it. *)
    let destroy =
      foreign "mlirDialectRegistryDestroy" (Typs.DialectRegistry.t @-> returning void)
  end

  (*===----------------------------------------------------------------------===
   * Location API.
   *===----------------------------------------------------------------------===*)

  module Location = struct
    open Typs.Location

    (* Creates an File/Line/Column location owned by the given context. *)
    let file_line_col_get =
      foreign
        "mlirLocationFileLineColGet"
        (Typs.Context.t @-> Typs.StringRef.t @-> uint @-> uint @-> returning t)


    (* Creates a call site location with a callee and a caller. *)
    let call_site = foreign "mlirLocationCallSiteGet" (t @-> t @-> returning t)

    (* Creates a fused location with an array of locations and metadata. *)
    let fused =
      foreign
        "mlirLocationFusedGet"
        (Typs.Context.t @-> intptr_t @-> ptr t @-> Typs.Attribute.t @-> returning t)


    (* Creates a name location owned by the given context. Providing null location
       for childLoc is allowed and if childLoc is null location, then the behavior
       is the same as having unknown child location. *)
    let name =
      foreign
        "mlirLocationNameGet"
        (Typs.Context.t @-> Typs.StringRef.t @-> t @-> returning t)


    (* Creates a location with unknown position owned by the given context. *)
    let unknown = foreign "mlirLocationUnknownGet" (Typs.Context.t @-> returning t)

    (* Gets the context that a location was created with. *)
    let context = foreign "mlirLocationGetContext" (t @-> returning Typs.Context.t)

    (* Checks if the location is null. *)
    let is_null = foreign "mlirLocationIsNull" (t @-> returning bool)

    (* Checks if two locations are equal. *)
    let equal = foreign "mlirLocationEqual" (t @-> t @-> returning bool)

    (* Prints a location by sending chunks of the string representation and
       forwarding `userData to `callback`. Note that the callback may be called
       several times with consecutive chunks of the string. *)
    let print =
      foreign
        "mlirLocationPrint"
        (t @-> Typs.string_callback @-> ptr void @-> returning void)
  end

  (*===----------------------------------------------------------------------===
   * Module API.
   *===----------------------------------------------------------------------===*)

  module Module = struct
    (* Creates a new, empty module and transfers ownership to the caller. *)
    let empty =
      foreign "mlirModuleCreateEmpty" (Typs.Location.t @-> returning Typs.Module.t)


    (* Parses a module from the string and transfers ownership to the caller. *)
    let parse =
      foreign
        "mlirModuleCreateParse"
        (Typs.Context.t @-> Typs.StringRef.t @-> returning Typs.Module.t)


    (* Gets the context that a module was created with. *)
    let context =
      foreign "mlirModuleGetContext" (Typs.Module.t @-> returning Typs.Context.t)


    (* Gets the body of the module, i.e. the only block it contains. *)
    let body = foreign "mlirModuleGetBody" (Typs.Module.t @-> returning Typs.Block.t)

    (* Checks whether a module is null. *)
    let is_null = foreign "mlirModuleIsNull" (Typs.Module.t @-> returning bool)

    (* Takes a module owned by the caller and deletes it. *)
    let destroy = foreign "mlirModuleDestroy" (Typs.Module.t @-> returning void)

    (* Views the generic operation as a module.
       The returned module is null when the input operation was not a ModuleOp. *)
    let operation =
      foreign "mlirModuleGetOperation" (Typs.Module.t @-> returning Typs.Operation.t)


    (* Views the module as a generic operation. *)
    let from_op =
      foreign "mlirModuleFromOperation" (Typs.Operation.t @-> returning Typs.Module.t)
  end

  (*===----------------------------------------------------------------------===
   * OperationState API.
   *===----------------------------------------------------------------------===*)

  module OperationState = struct
    open Typs.OperationState

    (* Constructs an operation state from a name and a location. *)
    let get =
      foreign
        "mlirOperationStateGet"
        (Typs.StringRef.t @-> Typs.Location.t @-> returning t)


    (* Adds a list of components to the operation state. *)
    let add_results =
      foreign
        "mlirOperationStateAddResults"
        (ptr t @-> intptr_t @-> ptr Typs.Type.t @-> returning void)


    let add_operands =
      foreign
        "mlirOperationStateAddOperands"
        (ptr t @-> intptr_t @-> ptr Typs.Value.t @-> returning void)


    let add_owned_regions =
      foreign
        "mlirOperationStateAddOwnedRegions"
        (ptr t @-> intptr_t @-> ptr Typs.Region.t @-> returning void)


    let add_successors =
      foreign
        "mlirOperationStateAddSuccessors"
        (ptr t @-> intptr_t @-> ptr Typs.Block.t @-> returning void)


    let add_attributes =
      foreign
        "mlirOperationStateAddAttributes"
        (ptr t @-> intptr_t @-> ptr Typs.NamedAttribute.t @-> returning void)


    (* Enables result type inference for the operation under construction. If
       enabled, then the caller must not have called
       mlirOperationStateAddResults(). Note that if enabled, the
       mlirOperationCreate() call is failable: it will return a null operation
       on inference failure and will emit diagnostics. *)
    let enable_result_type_inference =
      foreign "mlirOperationStateEnableResultTypeInference" (ptr t @-> returning void)
  end

  (*===----------------------------------------------------------------------===
   * OpPrintFlags API.
   *===----------------------------------------------------------------------===*)

  module OpPrintingFlags = struct
    open Typs.OpPrintingFlags

    (* Creates new printing flags with defaults, intended for customization.
       Must be freed with a call to mlirOpPrintingFlagsDestroy(). *)
    let create =
      foreign "mlirOpPrintingFlagsCreate" (void @-> returning Typs.OpPrintingFlags.t)


    (* Destroys printing flags created with mlirOpPrintingFlagsCreate. *)
    let destroy = foreign "mlirOpPrintingFlagsDestroy" (t @-> returning void)

    (* Enables the elision of large elements attributes by printing a lexically
       valid but otherwise meaningless form instead of the element data. The
       `largeElementLimit` is used to configure what is considered to be a "large"
       ElementsAttr by providing an upper limit to the number of elements. *)
    let large_element_limit =
      foreign
        "mlirOpPrintingFlagsElideLargeElementsAttrs"
        (t @-> intptr_t @-> returning void)


    (* Enable or disable printing of debug information (based on `enable`). If
       'prettyForm' is set to true, debug information is printed in a more readable
       'pretty' form. Note: The IR generated with 'prettyForm' is not parsable. *)
    let enable_debug_info =
      foreign "mlirOpPrintingFlagsEnableDebugInfo" (t @-> bool @-> bool @-> returning void)


    (* Always print operations in the generic form. *)
    let print_generic_op_form =
      foreign "mlirOpPrintingFlagsPrintGenericOpForm" (t @-> returning void)


    (* Use local scope when printing the operation. This allows for using the
       printer in a more localized and thread-safe setting, but may not
       necessarily be identical to what the IR will look like when dumping
       the full module. *)
    let use_local_scope = foreign "mlirOpPrintingFlagsUseLocalScope" (t @-> returning void)
  end

  (*===----------------------------------------------------------------------===
   * Operation API.
   *===----------------------------------------------------------------------===*)

  module Operation = struct
    (* Creates an operation and transfers ownership to the caller.
       Note that caller owned child objects are transferred in this call and must
       not be further used. Particularly, this applies to any regions added to
       the state (the implementation may invalidate any such pointers).

       This call can fail under the following conditions, in which case, it will
       return a null operation and emit diagnostics:
       - Result type inference is enabled and cannot be performed. *)
    let create =
      foreign
        "mlirOperationCreate"
        (ptr Typs.OperationState.t @-> returning Typs.Operation.t)


    (* Creates a deep copy of an operation. The operation is not inserted and
       ownership is transferred to the caller. *)
    let clone =
      foreign "mlirOperationClone" (Typs.Operation.t @-> returning Typs.Operation.t)


    (* Takes an operation owned by the caller and destroys it. *)
    let destroy = foreign "mlirOperationDestroy" (Typs.Operation.t @-> returning void)

    (* Removes the given operation from its parent block. The operation is not
       destroyed. The ownership of the operation is transferred to the caller. *)
    let remove_from_parent =
      foreign "mlirOperationRemoveFromParent" (Typs.Operation.t @-> returning void)


    (* Checks whether the underlying operation is null. *)
    let is_null = foreign "mlirOperationIsNull" (Typs.Operation.t @-> returning bool)

    (* Checks whether two operation handles point to the same operation. This does
       not perform deep comparison. *)
    let equal =
      foreign
        "mlirOperationEqual"
        (Typs.Operation.t @-> Typs.Operation.t @-> returning bool)


    (* Gets the context this operation is associated with *)
    let context =
      foreign "mlirOperationGetContext" (Typs.Operation.t @-> returning Typs.Context.t)


    (* Gets the location of the operation *)
    let loc =
      foreign "mlirOperationGetLocation" (Typs.Operation.t @-> returning Typs.Location.t)


    (* Gets the type id of the operation.
       Returns null if the operation does not have a registered operation
       description. *)
    let type_id =
      foreign "mlirOperationGetTypeID" (Typs.Operation.t @-> returning Typs.TypeID.t)


    (* Gets the name of the operation as an identifier. *)
    let name =
      foreign "mlirOperationGetName" (Typs.Operation.t @-> returning Typs.Identifier.t)


    (* Gets the block that owns this operation, returning null if the operation is
       not owned. *)
    let block =
      foreign "mlirOperationGetBlock" (Typs.Operation.t @-> returning Typs.Block.t)


    (* Gets the operation that owns this operation, returning null if the operation
       is not owned. *)
    let parent =
      foreign
        "mlirOperationGetParentOperation"
        (Typs.Operation.t @-> returning Typs.Operation.t)


    (* Returns the number of regions attached to the given operation. *)
    let num_regions =
      foreign "mlirOperationGetNumRegions" (Typs.Operation.t @-> returning intptr_t)


    (* Returns `pos`-th region attached to the operation. *)
    let region =
      foreign
        "mlirOperationGetRegion"
        (Typs.Operation.t @-> intptr_t @-> returning Typs.Region.t)


    (* Returns an operation immediately following the given operation it its
       enclosing block. *)
    let next_in_block =
      foreign
        "mlirOperationGetNextInBlock"
        (Typs.Operation.t @-> returning Typs.Operation.t)


    (* Returns the number of operands of the operation. *)
    let num_operands =
      foreign "mlirOperationGetNumOperands" (Typs.Operation.t @-> returning intptr_t)


    (* Returns `pos`-th operand of the operation. *)
    let operand =
      foreign
        "mlirOperationGetOperand"
        (Typs.Operation.t @-> intptr_t @-> returning Typs.Value.t)


    (* Sets the `pos`-th operand of the operation. *)
    let set_operand =
      foreign
        "mlirOperationSetOperand"
        (Typs.Operation.t @-> intptr_t @-> Typs.Value.t @-> returning void)


    (* Returns the number of results of the operation. *)
    let num_results =
      foreign "mlirOperationGetNumResults" (Typs.Operation.t @-> returning intptr_t)


    (* Returns `pos`-th result of the operation. *)
    let result =
      foreign
        "mlirOperationGetResult"
        (Typs.Operation.t @-> intptr_t @-> returning Typs.Value.t)


    (* Returns the number of successor blocks of the operation. *)
    let num_successors =
      foreign "mlirOperationGetNumSuccessors" (Typs.Operation.t @-> returning intptr_t)


    (* Returns `pos`-th successor of the operation. *)
    let succesor =
      foreign
        "mlirOperationGetSuccessor"
        (Typs.Operation.t @-> intptr_t @-> returning Typs.Block.t)


    (* Returns the number of attributes attached to the operation. *)
    let num_attributes =
      foreign "mlirOperationGetNumAttributes" (Typs.Operation.t @-> returning intptr_t)


    (* Return `pos`-th attribute of the operation. *)
    let attribute =
      foreign
        "mlirOperationGetAttribute"
        (Typs.Operation.t @-> intptr_t @-> returning Typs.NamedAttribute.t)


    (* Returns an attribute attached to the operation given its name. *)
    let attribute_by_name =
      foreign
        "mlirOperationGetAttributeByName"
        (Typs.Operation.t @-> Typs.StringRef.t @-> returning Typs.Attribute.t)


    (* Sets an attribute by name, replacing the existing if it exists or
       adding a new one otherwise. *)
    let set_attribute_by_name =
      foreign
        "mlirOperationSetAttributeByName"
        (Typs.Operation.t @-> Typs.StringRef.t @-> Typs.Attribute.t @-> returning void)


    (* Removes an attribute by name. Returns false if the attribute was not found
       and true if removed. *)
    let remove_attribute_by_name =
      foreign
        "mlirOperationRemoveAttributeByName"
        (Typs.Operation.t @-> Typs.StringRef.t @-> returning bool)


    (* Prints an operation by sending chunks of the string representation and
       forwarding `userData to `callback`. Note that the callback may be called
       several times with consecutive chunks of the string. *)
    let print =
      foreign
        "mlirOperationPrint"
        (Typs.Operation.t @-> Typs.string_callback @-> ptr void @-> returning void)


    (* Same as mlirOperationPrint but accepts flags controlling the printing
       behavior. *)
    let print_with_flags =
      foreign
        "mlirOperationPrintWithFlags"
        (Typs.Operation.t
         @-> Typs.OpPrintingFlags.t
         @-> Typs.string_callback
         @-> ptr void
         @-> returning void)


    (* Same as mlirOperationPrint but writing the bytecode format out. *)
    let write_bytecode =
      foreign
        "mlirOperationWriteBytecode"
        (Typs.Operation.t @-> Typs.string_callback @-> ptr void @-> returning void)


    (* Prints an operation to stderr. *)
    let dump = foreign "mlirOperationDump" (Typs.Operation.t @-> returning void)

    (* Verify the operation and return true if it passes, false if it fails. *)
    let verify = foreign "mlirOperationVerify" (Typs.Operation.t @-> returning bool)

    (* Moves the given operation immediately after the other operation in its
       parent block. The given operation may be owned by the caller or by its
       current block. The other operation must belong to a block. In any case, the
       ownership is transferred to the block of the other operation. *)
    let move_after =
      foreign
        "mlirOperationMoveAfter"
        (Typs.Operation.t @-> Typs.Operation.t @-> returning void)


    (* Moves the given operation immediately before the other operation in its
       parent block. The given operation may be owner by the caller or by its
       current block. The other operation must belong to a block. In any case, the
       ownership is transferred to the block of the other operation. *)
    let move_before =
      foreign
        "mlirOperationMoveBefore"
        (Typs.Operation.t @-> Typs.Operation.t @-> returning void)
  end

  module Region = struct
    (* Creates a new empty region and transfers ownership to the caller. *)
    let create = foreign "mlirRegionCreate" (void @-> returning Typs.Region.t)

    (* Takes a region owned by the caller and destroys it. *)
    let destroy = foreign "mlirRegionDestroy" (Typs.Region.t @-> returning void)

    (* Checks whether a region is null. *)
    let is_null = foreign "mlirRegionIsNull" (Typs.Region.t @-> returning bool)

    (* Checks whether two region handles point to the same region. This does not
       perform deep comparison. *)
    let equal =
      foreign "mlirRegionEqual" (Typs.Region.t @-> Typs.Region.t @-> returning bool)


    (* Gets the first block in the region. *)
    let first_block =
      foreign "mlirRegionGetFirstBlock" (Typs.Region.t @-> returning Typs.Block.t)


    (* Takes a block owned by the caller and appends it to the given region. *)
    let append_owned_block =
      foreign
        "mlirRegionAppendOwnedBlock"
        (Typs.Region.t @-> Typs.Block.t @-> returning void)


    (* Takes a block owned by the caller and inserts it at `pos` to the given
       region. This is an expensive operation that linearly scans the region,
       prefer insertAfter/Before instead. *)
    let insert_owned_block =
      foreign
        "mlirRegionInsertOwnedBlock"
        (Typs.Region.t @-> intptr_t @-> Typs.Block.t @-> returning void)


    (* Takes a block owned by the caller and inserts it after the (non-owned)
       reference block in the given region. The reference block must belong to the
       region. If the reference block is null, prepends the block to the region. *)
    let insert_owned_block_after =
      foreign
        "mlirRegionInsertOwnedBlockAfter"
        (Typs.Region.t @-> Typs.Block.t @-> Typs.Block.t @-> returning void)


    (* Takes a block owned by the caller and inserts it before the (non-owned)
       reference block in the given region. The reference block must belong to the
       region. If the reference block is null, appends the block to the region. *)
    let insert_owned_block_before =
      foreign
        "mlirRegionInsertOwnedBlockBefore"
        (Typs.Region.t @-> Typs.Block.t @-> Typs.Block.t @-> returning void)


    (* Returns first region attached to the operation. *)
    let first_region =
      foreign "mlirOperationGetFirstRegion" (Typs.Operation.t @-> returning Typs.Region.t)


    (* Returns the region immediately following the given region in its parent
       operation. *)
    let next_in_operation =
      foreign "mlirRegionGetNextInOperation" (Typs.Region.t @-> returning Typs.Region.t)
  end

  (*===----------------------------------------------------------------------===
   * Block API.
   *===----------------------------------------------------------------------===*)

  module Block = struct
    (* Creates a new empty block with the given argument types and transfers
       ownership to the caller. *)
    let create =
      foreign
        "mlirBlockCreate"
        (intptr_t @-> ptr Typs.Type.t @-> ptr Typs.Location.t @-> returning Typs.Block.t)


    (* Takes a block owned by the caller and destroys it. *)
    let destroy = foreign "mlirBlockDestroy" (Typs.Block.t @-> returning void)

    (* Detach a block from the owning region and assume ownership. *)
    let detach = foreign "mlirBlockDetach" (Typs.Block.t @-> returning void)

    (* Checks whether a block is null. *)
    let is_null = foreign "mlirBlockIsNull" (Typs.Block.t @-> returning bool)

    (* Checks whether two blocks handles point to the same block. This does not
       perform deep comparison. *)
    let equal = foreign "mlirBlockEqual" (Typs.Block.t @-> Typs.Block.t @-> returning bool)

    (* Returns the closest surrounding operation that contains this block. *)
    let parent_op =
      foreign "mlirBlockGetParentOperation" (Typs.Block.t @-> returning Typs.Operation.t)


    (* Returns the region that contains this block. *)
    let parent_region =
      foreign "mlirBlockGetParentRegion" (Typs.Block.t @-> returning Typs.Region.t)


    (* Returns the block immediately following the given block in its parent
       region. *)
    let next_in_region =
      foreign "mlirBlockGetNextInRegion" (Typs.Block.t @-> returning Typs.Block.t)


    (* Returns the first operation in the block. *)
    let first_operation =
      foreign "mlirBlockGetFirstOperation" (Typs.Block.t @-> returning Typs.Operation.t)


    (* Returns the terminator operation in the block or null if no terminator. *)
    let terminator =
      foreign "mlirBlockGetTerminator" (Typs.Block.t @-> returning Typs.Operation.t)


    (* Takes an operation owned by the caller and appends it to the block. *)
    let append_owned_operation =
      foreign
        "mlirBlockAppendOwnedOperation"
        (Typs.Block.t @-> Typs.Operation.t @-> returning void)


    (* Takes an operation owned by the caller and inserts it as `pos` to the block.
       This is an expensive operation that scans the block linearly, prefer
       insertBefore/After instead. *)
    let insert_owned_operation =
      foreign
        "mlirBlockInsertOwnedOperation"
        (Typs.Block.t @-> intptr_t @-> Typs.Operation.t @-> returning void)


    (* Takes an operation owned by the caller and inserts it after the (non-owned)
       reference operation in the given block. If the reference is null, prepends
       the operation. Otherwise, the reference must belong to the block. *)
    let insert_owned_operation_after =
      foreign
        "mlirBlockInsertOwnedOperationAfter"
        (Typs.Block.t @-> Typs.Operation.t @-> Typs.Operation.t @-> returning void)


    (* Takes an operation owned by the caller and inserts it before the (non-owned)
       reference operation in the given block. If the reference is null, appends
       the operation. Otherwise, the reference must belong to the block. *)
    let insert_owned_operation_before =
      foreign
        "mlirBlockInsertOwnedOperationBefore"
        (Typs.Block.t @-> Typs.Operation.t @-> Typs.Operation.t @-> returning void)


    (* Returns the number of arguments of the block. *)
    let num_arguments =
      foreign "mlirBlockGetNumArguments" (Typs.Block.t @-> returning intptr_t)


    (* Appends an argument of the specified type to the block. Returns the newly
       added argument. *)
    let add_argument =
      foreign
        "mlirBlockAddArgument"
        (Typs.Block.t @-> Typs.Type.t @-> Typs.Location.t @-> returning Typs.Value.t)


    (* Returns `pos`-th argument of the block. *)
    let argument =
      foreign "mlirBlockGetArgument" (Typs.Block.t @-> intptr_t @-> returning Typs.Value.t)


    (* Prints a block by sending chunks of the string representation and
       forwarding `userData to `callback`. Note that the callback may be called
       several times with consecutive chunks of the string. *)
    let print =
      foreign
        "mlirBlockPrint"
        (Typs.Block.t @-> Typs.string_callback @-> ptr void @-> returning void)
  end

  (*===----------------------------------------------------------------------===
   * Value API.
   *===----------------------------------------------------------------------===*)

  module Value = struct
    (* Returns whether the value is null. *)
    let is_null = foreign "mlirValueIsNull" (Typs.Value.t @-> returning bool)

    (* Returns 1 if two values are equal, 0 otherwise. *)
    (* mlirValueEqual does not seem to be exported *)
    let equal = foreign "mlirValueEqual" (Typs.Value.t @-> Typs.Value.t @-> returning bool)

    (* Returns 1 if the value is a block argument, 0 otherwise. *)
    let is_block_argument =
      foreign "mlirValueIsABlockArgument" (Typs.Value.t @-> returning bool)


    (* Returns 1 if the value is an operation result, 0 otherwise. *)
    let is_op_result = foreign "mlirValueIsAOpResult" (Typs.Value.t @-> returning bool)

    (* Returns the block in which this value is defined as an argument. Asserts if
     * the value is not a block argument. *)
    let block_argument_get_owner =
      foreign "mlirBlockArgumentGetOwner" (Typs.Value.t @-> returning Typs.Block.t)


    (* Returns the position of the value in the argument list of its block. *)
    let block_argument_arg_num =
      foreign "mlirBlockArgumentGetArgNumber" (Typs.Value.t @-> returning intptr_t)


    (* Sets the type of the block argument to the given type. *)
    let block_argument_set_type =
      foreign "mlirBlockArgumentSetType" (Typs.Value.t @-> Typs.Type.t @-> returning void)


    (* Returns an operation that produced this value as its result. Asserts if the
     * value is not an op result. *)
    let op_result_get_owner =
      foreign "mlirOpResultGetOwner" (Typs.Value.t @-> returning Typs.Operation.t)


    (* Returns the position of the value in the list of results of the operation
       * that produced it. *)
    let op_result_get_result_num =
      foreign "mlirOpResultGetResultNumber" (Typs.Value.t @-> returning intptr_t)


    (* Returns the type of the value. *)
    let get_type = foreign "mlirValueGetType" (Typs.Value.t @-> returning Typs.Type.t)

    (* Prints the value to the standard error stream. *)
    let dump = foreign "mlirValueDump" (Typs.Value.t @-> returning void)

    (* Prints a value by sending chunks of the string representation and
       * forwarding `userData to `callback`. Note that the callback may be called
       * several times with consecutive chunks of the string. *)
    let print =
      foreign
        "mlirValuePrint"
        (Typs.Value.t @-> Typs.string_callback @-> ptr void @-> returning void)


    (* Returns an op operand representing the first use of the value, or a null op
       operand if there are no uses. *)
    let first_use =
      foreign "mlirValueGetFirstUse" (Typs.Value.t @-> returning Typs.OpOperand.t)
  end

  (*===----------------------------------------------------------------------===
   *  OpOperand API.
   *===----------------------------------------------------------------------===*)

  module OpOperand = struct
    (* Returns whether the op operand is null. *)
    let is_null = foreign "mlirOpOperandIsNull" (Typs.OpOperand.t @-> returning bool)

    (* Returns the owner operation of an op operand. *)
    let owner =
      foreign "mlirOpOperandGetOwner" (Typs.OpOperand.t @-> returning Typs.Operation.t)


    (* Returns the operand number of an op operand. *)
    let operand_number =
      foreign "mlirOpOperandGetOperandNumber" (Typs.OpOperand.t @-> returning uint)


    (* Returns an op operand representing the next use of the value, or a null op
       operand if there is no next use. *)
    let next_use =
      foreign "mlirOpOperandGetNextUse" (Typs.OpOperand.t @-> returning Typs.OpOperand.t)
  end

  (*===----------------------------------------------------------------------===
   * Type API.
   *===----------------------------------------------------------------------===*)

  module Type = struct
    (* Parses a type. The type is owned by the context. *)
    let parse =
      foreign
        "mlirTypeParseGet"
        (Typs.Context.t @-> Typs.StringRef.t @-> returning Typs.Type.t)


    (* Gets the context that a type was created with. *)
    let context = foreign "mlirTypeGetContext" (Typs.Type.t @-> returning Typs.Context.t)

    (* Gets the type ID of the type. *)
    let type_id = foreign "mlirTypeGetTypeID" (Typs.Type.t @-> returning Typs.TypeID.t)

    (* Checks whether a type is null. *)
    let is_null = foreign "mlirTypeIsNull" (Typs.Type.t @-> returning bool)

    (* Checks if two types are equal. *)
    let equal = foreign "mlirTypeEqual" (Typs.Type.t @-> Typs.Type.t @-> returning bool)

    (* Prints a location by sending chunks of the string representation and
       * forwarding `userData to `callback`. Note that the callback may be called
       * several times with consecutive chunks of the string. *)
    let print =
      foreign
        "mlirTypePrint"
        (Typs.Type.t @-> Typs.string_callback @-> ptr void @-> returning void)


    (* Prints the type to the standard error stream. *)
    let dump = foreign "mlirTypeDump" (Typs.Type.t @-> returning void)
  end

  (*===----------------------------------------------------------------------===
   * Attribute API.
   *===----------------------------------------------------------------------===*)

  module Attribute = struct
    (* Parses an attribute. The attribute is owned by the context. *)
    let parse =
      foreign
        "mlirAttributeParseGet"
        (Typs.Context.t @-> Typs.StringRef.t @-> returning Typs.Attribute.t)


    (* Gets the context that an attribute was created with. *)
    let context =
      foreign "mlirAttributeGetContext" (Typs.Attribute.t @-> returning Typs.Context.t)


    (* Gets the type of this attribute. *)
    let get_type =
      foreign "mlirAttributeGetType" (Typs.Attribute.t @-> returning Typs.Type.t)


    (* Gets the type id of the attribute. *)
    let get_type_id =
      foreign "mlirAttributeGetTypeID" (Typs.Attribute.t @-> returning Typs.TypeID.t)


    (* Checks whether an attribute is null. *)
    let is_null = foreign "mlirAttributeIsNull" (Typs.Attribute.t @-> returning bool)

    (* Checks if two attributes are equal. *)
    let equal =
      foreign
        "mlirAttributeEqual"
        (Typs.Attribute.t @-> Typs.Attribute.t @-> returning bool)


    (* Prints an attribute by sending chunks of the string representation and
       * forwarding `userData to `callback`. Note that the callback may be called
       * several times with consecutive chunks of the string. *)
    let print =
      foreign
        "mlirAttributePrint"
        (Typs.Attribute.t @-> Typs.string_callback @-> ptr void @-> returning void)


    (* Prints the attribute to the standard error stream. *)
    let dump = foreign "mlirAttributeDump" (Typs.Attribute.t @-> returning void)

    (* Associates an attribute with the name. Takes ownership of neither. *)
    let name =
      foreign
        "mlirNamedAttributeGet"
        (Typs.Identifier.t @-> Typs.Attribute.t @-> returning Typs.NamedAttribute.t)
  end

  (* Identifier API *)
  module Identifier = struct
    (* Gets an identifier with the given string value. *)
    let get =
      foreign
        "mlirIdentifierGet"
        (Typs.Context.t @-> Typs.StringRef.t @-> returning Typs.Identifier.t)


    (* Returns the context associated with this identifier *)
    let context =
      foreign "mlirIdentifierGetContext" (Typs.Identifier.t @-> returning Typs.Context.t)


    let equal =
      foreign
        "mlirIdentifierEqual"
        (Typs.Identifier.t @-> Typs.Identifier.t @-> returning bool)


    let to_string =
      foreign "mlirIdentifierStr" (Typs.Identifier.t @-> returning Typs.StringRef.t)
  end

  (*===----------------------------------------------------------------------===
     Symbol and SymbolTable API.
    ===----------------------------------------------------------------------===*)
  module SymbolTable = struct
    (* Returns the name of the attribute used to store symbol names compatible with
       symbol tables. *)
    let symbol_attr_name =
      foreign "mlirSymbolTableGetSymbolAttributeName" (void @-> returning Typs.StringRef.t)


    (* Returns the name of the attribute used to store symbol visibility. *)
    let visibility_attr_name =
      foreign
        "mlirSymbolTableGetVisibilityAttributeName"
        (void @-> returning Typs.StringRef.t)


    (* Creates a symbol table for the given operation. If the operation does not
       have the SymbolTable trait, returns a null symbol table. *)
    let create =
      foreign "mlirSymbolTableCreate" (Typs.Operation.t @-> returning Typs.SymbolTable.t)


    (* Returns true if the symbol table is null. *)
    let is_null = foreign "mlirSymbolTableIsNull" (Typs.SymbolTable.t @-> returning bool)

    (* Destroys the symbol table created with mlirSymbolTableCreate. This does not
       affect the operations in the table. *)
    let destroy = foreign "mlirSymbolTableDestroy" (Typs.SymbolTable.t @-> returning void)

    (* Looks up a symbol with the given name in the given symbol table and returns
       the operation that corresponds to the symbol. If the symbol cannot be found,
       returns a null operation. *)
    let lookup =
      foreign
        "mlirSymbolTableLookup"
        (Typs.SymbolTable.t @-> Typs.StringRef.t @-> returning Typs.Operation.t)


    (* Inserts the given operation into the given symbol table. The operation must
       have the symbol trait. If the symbol table already has a symbol with the
       same name, renames the symbol being inserted to ensure name uniqueness. Note
       that this does not move the operation itself into the block of the symbol
       table operation, this should be done separately. Returns the name of the
       symbol after insertion. *)
    let insert =
      foreign
        "mlirSymbolTableInsert"
        (Typs.SymbolTable.t @-> Typs.Operation.t @-> returning Typs.Attribute.t)


    (* Removes the given operation from the symbol table and erases it. *)
    let erase =
      foreign
        "mlirSymbolTableErase"
        (Typs.SymbolTable.t @-> Typs.Operation.t @-> returning void)


    (* Attempt to replace all uses that are nested within the given operation
       of the given symbol 'oldSymbol' with the provided 'newSymbol'. This does
       not traverse into nested symbol tables. Will fail atomically if there are
       any unknown operations that may be potential symbol tables. *)
    let replace_all_symbol_uses =
      foreign
        "mlirSymbolTableReplaceAllSymbolUses"
        (Typs.StringRef.t
         @-> Typs.StringRef.t
         @-> Typs.Operation.t
         @-> returning Typs.LogicalResult.t)


    (* Walks all symbol table operations nested within, and including, `op`. For
       each symbol table operation, the provided callback is invoked with the op
       and a boolean signifying if the symbols within that symbol table can be
       treated as if all uses within the IR are visible to the caller.
       `allSymUsesVisible` identifies whether all of the symbol uses of symbols
       within `op` are visible. *)
    let walk_symbol_tables =
      foreign
        "mlirSymbolTableWalkSymbolTables"
        (Typs.Operation.t
         @-> bool
         @-> Ctypes.(
               Foreign.funptr (Typs.Operation.t @-> bool @-> ptr void @-> returning void))
         @-> ptr void
         @-> returning void)
  end
end
