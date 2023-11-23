(** MLIR Context *)
type mlcontext

(** MLIR Dialect *)
type mldialect

(** MLIR Type *)
type mltype

(** MLIR Type ID *)
type mltypeid

(** MLIR Type ID Allocator *)
type mltypeid_alloc

(** MLIR Block *)
type mlblock

(** MLIR Region *)
type mlregion

(** MLIR Location *)
type mllocation

(** MLIR Module *)
type mlmodule

(** MLIR Value *)
type mlvalue

(** MLIR Operation *)
type mlop

(** MLIR Operation State *)
type mlop_state

(** MLIR Attribute *)
type mlattr

(** MLIR Named Attribute *)
type mlnamed_attr

(** MLIR Pass *)
type mlpass

(** MLIR Pass Manager *)
type mlpm

(** MLIR Op Pass Manager *)
type mlop_pm

(** MLIR Identifier *)
type mlident

(** MLIR Affine expr *)
type mlaffine_expr

(** MLIR Dialect Handle *)
type mldialect_handle = Stubs.Typs.DialectHandle.t Ctypes.structure

(** MLIR Dialect Registry *)
type mldialect_registry

(** MLIR Symbol Table *)
type mlsymboltbl

(** MLIR OpOperand *)
type mlop_operand

(** MLIR External Pass *)
type mlexternal_pass

(** MLIR External Pass Callbacks *)
type mlexternal_pass_callbacks =
  { construct : unit -> unit
  ; destruct : unit -> unit
  ; initialize : (mlcontext -> unit -> bool) option
  ; clone : unit -> unit
  ; run : mlop -> mlexternal_pass -> unit
  }

module IR : sig
  module Context : sig
    val global_ctx : mlcontext

    (** Creates an MLIR context and transfers its ownership to the caller. *)
    val create : unit -> mlcontext

    (** Checks if two contexts are equal. *)
    val equal : mlcontext -> mlcontext -> bool

    (** Takes an MLIR context owned by the caller and destroys it. *)
    val destroy : mlcontext -> unit

    (** Checks whether a context is null. *)
    val is_null : mlcontext -> bool

    (** Returns the number of dialects registered with the given context. A registered dialect will be loaded if needed by the parser. *)
    val num_registered_dialects : mlcontext -> int

    (** Returns the number of dialects loaded by the context.*)
    val num_loaded_dialects : mlcontext -> int

    (** Sets whether unregistered dialects are allowed in this context. *)
    val set_allow_unregistered_dialects : mlcontext -> bool -> unit

    (** Returns whether the context allows unregistered dialects. *)
    val get_allow_unregistered_dialects : mlcontext -> bool

    (** Gets the dialect instance owned by the given context using the dialect namespace to identify it, loads (i.e., constructs the instance of) the dialect if necessary. If the dialect is not registered with the context, returns null. Use mlirContextLoad<Name>mldialecto load an unregistered dialect. *)
    val get_or_load_dialect : mlcontext -> string -> mldialect

    (** Set threading mode (must be set to false to mlir-print-ir-after-all). *)
    val enable_multithreading : mlcontext -> bool -> unit
  end

  module Dialect : sig
    (** Returns the context that owns the dialect. *)
    val context : mldialect -> mlcontext

    (** Checks if the dialect is null. *)
    val is_null : mldialect -> bool

    (** Checks if two dialects that belong to the same context are equal. Dialects from different contexts will not compare equal. *)
    val equal : mldialect -> mldialect -> bool

    (** Returns the namespace of the given dialect. *)
    val namespace : mldialect -> string
  end

  module DialectHandle : sig
    (** Returns the namespace associated with the provided dialect handle. *)
    val namespace : mldialect_handle -> string

    (** Inserts the dialect associated with the provided dialect handle into the
        provided dialect registry *)
    val insert : mldialect_handle -> mldialect_registry -> unit

    (** Registers the dialect associated with the provided dialect handle. *)
    val register : mldialect_handle -> mlcontext -> unit

    (** Loads the dialect associated with the provided dialect handle. *)
    val load : mldialect_handle -> mlcontext -> mldialect
  end

  module DialectRegistry : sig
    (** Creates a dialect registry and transfers its ownership to the caller. *)
    val create : unit -> mldialect_registry

    (** Checks if the dialect registry is null. *)
    val is_null : mldialect_registry -> bool

    (** Takes a dialect registry owned by the caller and destroys it. *)
    val destroy : mldialect_registry -> unit
  end

  module Type : sig
    (** Parses a type. The type is owned by the context. *)
    val parse : mlcontext -> string -> mltype

    (** Gets the context that a type was created with. *)
    val context : mltype -> mlcontext

    (** Checks whether a type is null. *)
    val is_null : mltype -> bool

    (** Checks if two types are equal. *)
    val equal : mltype -> mltype -> bool

    (** Prints a location by sending chunks of the string representation and forwarding `userData to `callback`. Note that the callback may be called several times with consecutive chunks of the string. *)
    val print : callback:(string -> unit) -> mltype -> unit

    (** Prints the type to the standard error stream. *)
    val dump : mltype -> unit
  end

  module Region : sig
    (** Creates a new empty region and transfers ownership to the caller. *)
    val create : unit -> mlregion

    (** Takes a region owned by the caller and destroys it. *)
    val destroy : mlregion -> unit

    (** Checks whether a region is null. *)
    val is_null : mlregion -> bool

    (** Gets the first block in the region. *)
    val first_block : mlregion -> mlblock

    (** Takes a block owned by the caller and appends it to the given region. *)
    val append_owned_block : mlregion -> mlblock -> unit

    (** Takes a block owned by the caller and inserts it at `pos` to the given region. This is an expensive operation that linearly scans the region, prefer insertAfter/Before instead. *)
    val insert_owned_block_before : mlregion -> mlblock -> mlblock -> unit

    (** Takes a block owned by the caller and inserts it after the (non-owned) reference block in the given region. The reference block must belong to the region. If the reference block is null, prepends the block to the region. *)
    val insert_owned_block_after : mlregion -> mlblock -> mlblock -> unit
  end

  module Location : sig
    (** Creates an File/Line/Column location owned by the given context. *)
    val file_line_col_get : mlcontext -> string -> int -> int -> mllocation

    (** Creates a location with unknown position owned by the given context. *)
    val unknown : mlcontext -> mllocation

    (** Gets the context that a location was created with. *)
    val context : mllocation -> mlcontext

    (** Checks if the location is null. *)
    val is_null : mllocation -> bool

    (** Checks if two locations are equal. *)
    val equal : mllocation -> mllocation -> bool

    (** Prints a location by sending chunks of the string representation and forwarding `userData to `callback`. Note that the callback may be called several times with consecutive chunks of the string. *)
    val print : callback:(string -> unit) -> mllocation -> unit
  end

  module Attribute : sig
    (** Parses an attribute. The attribute is owned by the context. *)
    val parse : mlcontext -> string -> mlattr

    (** Gets the context that an attribute was created with. *)
    val context : mlattr -> mlcontext

    (** Gets the type of this attribute. *)
    val get_type : mlattr -> mltype

    (** Checks whether an attribute is null. *)
    val is_null : mlattr -> bool

    (** Checks if two attributes are equal. *)
    val equal : mlattr -> mlattr -> bool

    (** Prints an attribute by sending chunks of the string representation and forwarding `userData to `callback`. Note that the callback may be called several times with consecutive chunks of the string. *)
    val print : callback:(string -> unit) -> mlattr -> unit

    (** Prints the attribute to the standard error stream. *)
    val dump : mlattr -> unit

    (** Associates an attribute with the name. Takes ownership of neither. *)
    val name : mlident -> mlattr -> mlnamed_attr
  end

  module OperationState : sig
    (** Constructs an operation state from a name and a location. *)
    val get : string -> mllocation -> mlop_state

    (** Adds a list of results to the operation state. *)
    val add_results : mlop_state -> mltype list -> unit

    (** Adds a list of named attributes to the operation state. *)
    val add_named_attributes : mlop_state -> mlnamed_attr list -> unit

    (** Adds a list of regions to the operation state. *)
    val add_owned_regions : mlop_state -> mlregion list -> unit

    (** Adds a list of operands to the operation state. *)
    val add_operands : mlop_state -> mlvalue list -> unit
  end

  module Operation : sig
    (** Creates an operation and transfers ownership to the caller. *)
    val create : mlop_state -> mlop

    (** Creates a deep copy of an operation. The operation is not inserted and
        ownership is transferred to the caller. *)
    val clone : mlop -> mlop

    (** Takes an operation owned by the caller and destroys it. *)
    val destroy : mlop -> unit

    (** Removes the given operation from its parent block. The operation is not
        destroyed. The ownership of the operation is transferred to the caller. *)
    val remove_from_parent : mlop -> unit

    (** Checks whether the underlying operation is null. *)
    val is_null : mlop -> bool

    (** Checks whether two operation handles point to the same operation. This does
        not perform deep comparison. *)
    val equal : mlop -> mlop -> bool

    (** Gets the location of the operation. *)
    val loc : mlop -> mllocation

    (** Gets the name of the operation as an identifier. *)
    val name : mlop -> string

    (** Gets the block that owns this operation, returning null if the operation is
        not owned. *)
    val block : mlop -> mlblock

    (** Returns the number of regions attached to the given operation. *)
    val num_regions : mlop -> int

    (** Returns `pos`-th region attached to the operation. *)
    val region : mlop -> int -> mlregion

    (** Returns an operation immediately following the given operation it its
        enclosing block. *)
    val next_in_block : mlop -> mlop

    (** Returns the number of operands of the operation. *)
    val num_operands : mlop -> int

    (** Returns `pos`-th operand of the operation. *)
    val operand : mlop -> int -> mlvalue

    (** Sets the `pos`-th operand of the operation. *)
    val set_operand : mlop -> int -> mlvalue -> unit

    (** Returns the number of results of the operation. *)
    val num_results : mlop -> int

    (** Returns `pos`-th result of the operation. *)
    val result : mlop -> int -> mlvalue

    (** Returns an attribute attached to the operation given its name. *)
    val attribute_by_name : mlop -> string -> mlattr

    (** Sets an attribute by name, replacing the existing if it exists or
        adding a new one otherwise. *)
    val set_attribute_by_name : mlop -> string -> mlattr -> unit

    (** Prints an operation to stderr. *)
    val dump : mlop -> unit

    (** Verify the operation and return true if it passes, false if it fails. *)
    val verify : mlop -> bool
  end

  module Value : sig
    (** Returns whether the value is null. *)
    val is_null : mlvalue -> bool

    (** Returns 1 if two values are equal, 0 otherwise. *)
    val equal : mlvalue -> mlvalue -> bool

    (** Returns 1 if the value is a block argument, 0 otherwise. *)
    val is_block_argument : mlvalue -> bool

    (** Returns 1 if the value is an operation result, 0 otherwise. *)
    val is_op_result : mlvalue -> bool

    (** Returns the block in which this value is defined as an argument. Asserts if the value is not a block argument. *)
    val block_argument_get_owner : mlvalue -> mlblock

    (** Returns the position of the value in the argument list of its block. *)
    val block_argument_arg_num : mlvalue -> int

    (** Sets the type of the block argument to the given type. *)
    val block_argument_set_type : mlvalue -> mltype -> unit

    (** Returns an operation that produced this value as its result. Asserts if the value is not an op result. *)
    val op_result_get_owner : mlvalue -> mlop

    (** Returns the position of the value in the list of results of the operation that produced it. *)
    val op_result_get_result_num : mlvalue -> int

    (** Returns the type of the value. *)
    val get_type : mlvalue -> mltype

    (** Prints the value to the standard error stream. *)
    val dump : mlvalue -> unit

    (** Prints a value by sending chunks of the string representation and forwarding `userData to `callback`. Note that the callback may be called several times with consecutive chunks of the string. *)
    val print : callback:(string -> unit) -> mlvalue -> unit

    (** Returns an op operand representing the first use of the value, or a null op
        operand if there are no uses. *)
    val first_use : mlvalue -> mlop_operand

    (** Replace all uses of the [old] value with [fresh]. *)
    val replace_uses : old:mlvalue -> fresh:mlvalue -> unit
  end

  module OpOperand : sig
    (** Returns whether the op operand is null. *)
    val is_null : mlop_operand -> bool

    (** Returns the owner operation of an op operand. *)
    val owner : mlop_operand -> mlop

    (** Returns the operand number of an op operand. *)
    val operand_number : mlop_operand -> int

    (** Returns an op operand representing the next use of the value, or a null op
        operand if there is no next use. *)
    val next_use : mlop_operand -> mlop_operand
  end

  module Block : sig
    (** Creates a new empty block with the given argument types and transfers ownership to the caller. *)
    val create : mltype list -> mllocation -> mlblock

    (** Takes a block owned by the caller and destroys it. *)
    val destroy : mlblock -> unit

    (** Detach a block from the owning region and assume ownership. *)
    val detach : mlblock -> unit

    (** Checks whether a block is null. *)
    val is_null : mlblock -> bool

    (** Returns the first operation in the block. *)
    val first_operation : mlblock -> mlop

    (** Takes an operation owned by the caller and inserts it as `pos` to the block. This is an expensive operation that scans the block linearly, prefer insertBefore/After instead. *)
    val insert_owned_operation : mlblock -> int -> mlop -> unit

    (** Takes an operation owned by the caller and inserts it before the (non-owned) reference operation in the given block. If the reference is null, appends the operation. Otherwise, the reference must belong to the block. *)
    val insert_owned_operation_before : mlblock -> mlop -> mlop -> unit

    (** Takes an operation owned by the caller and inserts it after the (non-owned) reference operation in the given block. If the reference is null, prepends the operation. Otherwise, the reference must belong to the block. *)
    val insert_owned_operation_after : mlblock -> mlop -> mlop -> unit

    (** Returns the terminator operation in the block or null if no terminator. *)
    val terminator : mlblock -> mlop

    (** Takes an operation owned by the caller and appends it to the block. *)
    val append_owned_operation : mlblock -> mlop -> unit

    (** Returns the number of arguments of the block. *)
    val num_arguments : mlblock -> int

    (** Appends an argument of the specified type to the block. Returns the newly
        added argument. *)
    val add_argument : mlblock -> mltype -> mllocation -> mlvalue

    (** Returns `pos`-th argument of the block. *)
    val argument : mlblock -> int -> mlvalue

    (** Returns operations in the block. *)
    val ops : mlblock -> mlop list
  end

  module Module : sig
    (** Creates a new, empty module and transfers ownership to the caller. *)
    val empty : mllocation -> mlmodule

    (** Parses a module from the string and transfers ownership to the caller. *)
    val parse : mlcontext -> string -> mlmodule

    (** Checks whether a module is null. *)
    val is_null : mlmodule -> bool

    (** Takes a module owned by the caller and deletes it. *)
    val destroy : mlmodule -> unit

    (** Gets the context that a module was created with. *)
    val context : mlmodule -> mlcontext

    (** Gets the body of the module, i.e. the only block it contains. *)
    val body : mlmodule -> mlblock

    (** Views the module as a generic operation. *)
    val operation : mlmodule -> mlop
  end

  module Identifier : sig
    (** Gets an identifier with the given string value. *)
    val get : mlcontext -> string -> mlident

    (** Returns the context associated with this identifier. *)
    val context : mlident -> mlcontext

    (** Checks whether two identifiers are the same. *)
    val equal : mlident -> mlident -> bool

    (** Gets the string value of the identifier. *)
    val to_string : mlident -> string
  end

  module SymbolTable : sig
    (** Returns the name of the attribute used to store symbol names compatible with
        symbol tables. *)
    val symbol_attr_name : unit -> string

    (** Returns the name of the attribute used to store symbol visibility. *)
    val visibility_attr_name : unit -> string

    (** Creates a symbol table for the given operation. If the operation does not
        have the SymbolTable trait, returns a null symbol table. *)
    val create : mlop -> mlsymboltbl

    (** Returns true if the symbol table is null. *)
    val is_null : mlsymboltbl -> bool

    (** Destroys the symbol table created with mlirSymbolTableCreate. This does not
        affect the operations in the table. *)
    val destroy : mlsymboltbl -> unit

    (** Looks up a symbol with the given name in the given symbol table and returns
        the operation that corresponds to the symbol. If the symbol cannot be found,
        returns a null operation. *)
    val lookup : mlsymboltbl -> string -> mlop

    (** Inserts the given operation into the given symbol table. The operation must
        have the symbol trait. If the symbol table already has a symbol with the
        same name, renames the symbol being inserted to ensure name uniqueness. Note
        that this does not move the operation itself into the block of the symbol
        table operation, this should be done separately. Returns the name of the
        symbol after insertion. *)
    val insert : mlsymboltbl -> mlop -> mlattr

    (** Removes the given operation from the symbol table and erases it. *)
    val erase : mlsymboltbl -> mlop -> unit

    (** Attempt to replace all uses that are nested within the given operation
        of the given symbol 'oldSymbol' with the provided 'newSymbol'. This does
        not traverse into nested symbol tables. Will fail atomically if there are
        any unknown operations that may be potential symbol tables. *)
    val replace_all_symbol_uses : old_sym:string -> new_sym:string -> mlop -> bool
  end
end

module AffineExpr : sig
  type t

  (** Gets the context that owns the affine expression. *)
  val context : t -> mlcontext

  (** Prints an affine expression by sending chunks of the string representation and forwarding `userData to `callback`. Note that the callback may be called several times with consecutive chunks of the string. *)
  val print : callback:(string -> unit) -> t -> unit

  (** Prints the affine expression to the standard error stream. *)
  val dump : t -> unit

  (** Checks whether the given affine expression is made out of only symbols and constants. *)
  val is_symbolic_or_constant : t -> bool

  (** Checks whether the given affine expression is a pure affine expression, i.e. mul, floordiv, ceildic, and mod is only allowed w.r.t constants. *)
  val is_pure_affine : t -> bool

  (** Returns the greatest known integral divisor of this affine expression. The result is always positive. *)
  val largest_known_divisor : t -> int

  (** Checks whether the given affine expression is a multiple of 'factor'. *)
  val is_multiple_of : t -> int -> bool

  (** Checks whether the given affine expression involves AffineDimExpr 'position'. *)
  val is_function_of_dim : t -> int -> bool

  module Dimension : sig
    (** Creates an affine dimension expression with 'position' in the context. *)
    val get : mlcontext -> int -> t

    (** Returns the position of the given affine dimension expression. *)
    val position : t -> int
  end

  module Symbol : sig
    (** Creates an affine symbol expression with 'position' in the context. *)
    val get : mlcontext -> int -> t

    (** Returns the position of the given affine symbol expression. *)
    val position : t -> int
  end

  module Constant : sig
    (** Creates an affine constant expression with 'constant' in the context. *)
    val get : mlcontext -> int -> t

    (** Returns the  of the given affine constant expression. *)
    val value : t -> int
  end

  module Add : sig
    (** Checks whether the given affine expression is an add expression. *)
    val is_add : t -> bool

    (** Creates an affine add expression with 'lhs' and 'rhs'. *)
    val get : t -> t -> t
  end

  module Mul : sig
    (** Checks whether the given affine expression is an mul expression. *)
    val is_mul : t -> bool

    (** Creates an affine mul expression with 'lhs' and 'rhs'. *)
    val get : t -> t -> t
  end

  module Mod : sig
    (** Checks whether the given affine expression is an mod expression. *)
    val is_mod : t -> bool

    (** Creates an affine mod expression with 'lhs' and 'rhs'. *)
    val get : t -> t -> t
  end

  module FloorDiv : sig
    (** Checks whether the given affine expression is an floordiv expression. *)
    val is_floor_div : t -> bool

    (** Creates an affine floordiv expression with 'lhs' and 'rhs'. *)
    val get : t -> t -> t
  end

  module CeilDiv : sig
    (** Checks whether the given affine expression is an ceildiv expression. *)
    val is_ceil_div : t -> bool

    (** Creates an affine ceildiv expression with 'lhs' and 'rhs'. *)
    val get : t -> t -> t
  end

  module BinaryOp : sig
    (** Returns the left hand side affine expression of the given affine binary operation expression. *)
    val lhs : t -> t

    (** Returns the right hand side affine expression of the given affine binary operation expression. *)
    val rhs : t -> t
  end
end

module AffineMap : sig
  type t

  (** Gets the context that the given affine map was created with *)
  val context : t -> mlcontext

  (** Checks whether an affine map is null. *)
  val is_null : t -> bool

  (** Checks if two affine maps are equal. *)
  val equal : t -> t -> bool

  (** Prints an affine map by sending chunks of the string representation and forwarding `userData to `callback`. Note that the callback may be called several times with consecutive chunks of the string. *)
  val print : callback:(string -> unit) -> t -> unit

  (** Prints the affine map to the standard error stream. *)
  val dump : t -> unit

  (** Creates a zero result affine map with no dimensions or symbols in the context. The affine map is owned by the context. *)
  val empty : mlcontext -> t

  (** Creates a zero result affine map of the given dimensions and symbols in the context. The affine map is owned by the context. *)
  val get : mlcontext -> int -> int -> int -> mlaffine_expr -> t

  (** Creates a single constant result affine map in the context. The affine map is owned by the context. *)
  val constant : mlcontext -> int -> t

  (** Creates an affine map with 'numDims' identity in the context. The affine map is owned by the context. *)
  val multi_dim_identity : mlcontext -> int -> t

  (** Creates an identity affine map on the most minor dimensions in the context. The affine map is owned by the context. The function asserts that the number of dimensions is greater or equal to the number of results. *)
  val minor_identity : mlcontext -> int -> int -> t

  (** Creates an affine map with a permutation expression and its size in the context. The permutation expression is a non-empty vector of integers. The elements of the permutation vector must be continuous from 0 and cannot be repeated (i.e. `[1,2,0]` is a valid permutation. `[2,0]` or `[1,1,2]` is an invalid invalid permutation.) The affine map is owned by the context. *)
  val permutation : mlcontext -> int list -> t

  (** Checks whether the given affine map is an identity affine map. The function asserts that the number of dimensions is greater or equal to the number of results. *)
  val is_identity : t -> bool

  (** Checks whether the given affine map is a minor identity affine map. *)
  val is_minor_identity : t -> bool

  (** Checks whether the given affine map is a empty affine map. *)
  val is_empty : t -> bool

  (** Checks whether the given affine map is a single result constant affine map. *)
  val is_single_constant : t -> bool

  (** Returns the constant result of the given affine map. The function asserts
      * that the map has a single constant result. *)
  val single_constant_result : t -> int

  (** Returns the number of dimensions of the given affine map. *)
  val num_dims : t -> int

  (** Returns the number of symbols of the given affine map. *)
  val num_symbols : t -> int

  (** Returns the number of results of the given affine map. *)
  val num_results : t -> int

  (** Returns the number of inputs (dimensions + symbols) of the given affine map. *)
  val num_inputs : t -> int

  (** Checks whether the given affine map represents a subset of a symbol-less permutation map. *)
  val is_projected_permutation : t -> bool

  (** Checks whether the given affine map represents a symbol-less permutation map. *)
  val is_permutation : t -> bool

  (** Returns the affine map consisting of the `resultPos` subset. *)
  val sub_map : t -> int list -> t

  (** Returns the affine map consisting of the most major `numResults` results. Returns the null AffineMap if the `numResults` is equal to zero. Returns the `affineMap` if `numResults` is greater or equals to number of results of the given affine map. *)
  val major_sub_map : t -> int -> t

  (** Returns the affine map consisting of the most minor `numResults` results. Returns the null AffineMap if the `numResults` is equal to zero. Returns the `affineMap` if `numResults` is greater or equals to number of results of the given affine map. *)
  val minor_sub_map : t -> int -> t
end

module BuiltinTypes : sig
  module Integer : sig
    (** Checks whether the given type is an integer type. *)
    val is_integer : mltype -> bool

    (** Creates a signless integer type of the given bitwidth in the context. The type is owned by the context. *)
    val get : mlcontext -> int -> mltype

    (** Creates a signed integer type of the given bitwidth in the context. The type is owned by the context. *)
    val signed : mlcontext -> int -> mltype

    (** Creates an unsigned integer type of the given bitwidth in the context. The type is owned by the context. *)
    val unsigned : mlcontext -> int -> mltype

    (** Returns the bitwidth of an integer type. *)
    val width : mltype -> int

    (** Checks whether the given integer type is signless. *)
    val is_signless : mltype -> bool

    (** Checks whether the given integer type is signed. *)
    val is_signed : mltype -> bool

    (** Checks whether the given integer type is unsigned. *)
    val is_unsigned : mltype -> bool
  end

  module Float : sig
    (** Checks whether the given type is a bf16 type. *)
    val is_bf16 : mltype -> bool

    (** Creates a bf16 type in the given context. The type is owned by the context. *)
    val bf16 : mlcontext -> mltype

    (** Checks whether the given type is an f16 type. *)
    val is_f16 : mltype -> bool

    (** Creates an f16 type in the given context. The type is owned by the context. *)
    val f16 : mlcontext -> mltype

    (** Checks whether the given type is an f32 type. *)
    val is_f32 : mltype -> bool

    (** Creates an f32 type in the given context. The type is owned by the context. *)
    val f32 : mlcontext -> mltype

    (** Checks whether the given type is an f64 type. *)
    val is_f64 : mltype -> bool

    (** Creates a f64 type in the given context. The type is owned by the context. *)
    val f64 : mlcontext -> mltype
  end

  module Index : sig
    (** Checks whether the given type is an index type. *)
    val is_index : mltype -> bool

    (** Creates an index type in the given context. The type is owned by the context. *)
    val get : mlcontext -> mltype
  end

  module None : sig
    (** Checks whether the given type is a None type. *)
    val is_none : mltype -> bool

    (** Creates a None type in the given context. The type is owned by the context. *)
    val get : mlcontext -> mltype
  end

  module Complex : sig
    (** Checks whether the given type is a Complex type. *)
    val is_complex : mltype -> bool

    (** Creates a complex type with the given element type in the same context as the element type. The type is owned by the context. *)
    val get : mltype -> mltype

    (** Returns the element type of the given complex type. *)
    val element_type : mltype -> mltype
  end

  module Shaped : sig
    (** Checks whether the given type is a Shaped type. *)
    val is_shaped : mltype -> bool

    (** Returns the element type of the shaped type. *)
    val element_type : mltype -> mltype

    (** Checks whether the given shaped type is ranked. *)
    val has_rank : mltype -> bool

    (** Returns the rank of the given ranked shaped type. *)
    val rank : mltype -> int

    (** Checks whether the given shaped type has a static shape. *)
    val has_static_shape : mltype -> bool

    (** Checks wither the dim-th dimension of the given shaped type is dynamic. *)
    val is_dynamic_dim : mltype -> int -> bool

    (** Returns the dim-th dimension of the given ranked shaped type. *)
    val dim_size : mltype -> int -> int

    (** Checks whether the given value is used as a placeholder for dynamic sizes
        in shaped types. *)
    val is_dynamic_size : int -> bool

    (** Returns the value indicating a dynamic size in a shaped type. Prefer
        mlirShapedTypeIsDynamicSize to direct comparisons with this value. *)
    val dynamic_size : unit -> int

    (** Checks whether the given value is used as a placeholder for dynamic strides
        and offsets in shaped types. *)
    val is_dynamic_stride_or_offset : int -> bool

    (** Returns the value indicating a dynamic stride or offset in a shaped type.
        Prefer mlirShapedTypeGetDynamicStrideOrOffset to direct comparisons with
        this value. *)
    val dynamic_stride_or_offset : unit -> int
  end

  module Vector : sig
    (** Checks whether the given type is a Vector type. *)
    val is_vector : mltype -> bool

    (** Creates a vector type of the shape identified by its rank and dimensions, with the given element type in the same context as the element type. The type is owned by the context. *)
    val get : int array -> mltype -> mltype

    (** Same as "mlirVectorTypeGet" but returns a nullptr wrapping MlirType on illegal arguments, emitting appropriate diagnostics. *)
    val get_checked : mllocation -> int -> int array -> mltype -> mltype
  end

  module Tensor : sig
    (** Checks whether the given type is a Tensor type. *)
    val is_tensor : mltype -> bool

    (** Checks whether the given type is a ranked tensor type. *)
    val is_ranked_tensor : mltype -> bool

    (** Checks whether the given type is an unranked tensor type. *)
    val is_unranked_tensor : mltype -> bool

    (** Creates a tensor type of a fixed rank with the given shape and element type in the same context as the element type. The type is owned by the context. *)
    val ranked : int array -> mltype -> mlattr -> mltype

    (** Same as "mlirRankedTensorTypeGet" but returns a nullptr wrapping MlirType on illegal arguments, emitting appropriate diagnostics. *)
    val ranked_checked : mllocation -> int array -> mltype -> mlattr -> mltype

    (** Creates an unranked tensor type with the given element type in the same context as the element type. The type is owned by the context. *)
    val unranked : mltype -> mltype

    (** Same as "mlirUnrankedTensorTypeGet" but returns a nullptr wrapping MlirType on illegal arguments, emitting appropriate diagnostics. *)
    val unranked_checked : mllocation -> mltype -> mltype
  end

  module MemRef : sig
    (** Checks whether the given type is a MemRef type. *)
    val is_memref : mltype -> bool

    (** Checks whether the given type is an UnrankedMemRef type. *)
    val is_unranked_memref : mltype -> bool

    (** Creates a MemRef type with the given shape, a potentially empty list of affine layout maps, the given memory space and element type, in the same context as element type. The type is owned by the context. *)
    val get : mltype -> int array -> mlattr -> mlattr -> mltype

    (** Same as `get` but returns a nullptr-wrapping MlirType o
        illegal arguments, emitting appropriate diagnostics. *)
    val checked : mllocation -> mltype -> int array -> mlattr -> mlattr -> mltype

    (** Creates a MemRef type with the given shape, memory space and element type in the same context as the element type. The type has no affine maps, i.e. represents a default row-major contiguous memref. The type is owned by the context. *)
    val contiguous : mltype -> int array -> mlattr -> mltype

    (** Same as "mlirMemRefTypeContiguousGet" but returns a nullptr wrapping MlirType on illegal arguments, emitting appropriate diagnostics. *)
    val contiguous_checked : mllocation -> mltype -> int array -> mlattr -> mltype

    (** Creates an Unranked MemRef type with the given element type and in the given memory space. The type is owned by the context of element type. *)
    val unranked : mltype -> mlattr -> mltype

    (** Same as `unranked` but returns a nullptr wrapping MlirType on illegal arguments, emitting appropriate diagnostics. *)
    val unranked_checked : mllocation -> mltype -> mlattr -> mltype

    (** Returns the layout of the given MemRef type. *)
    val layout : mltype -> mlattr

    (** Returns the pos-th affine map of the given MemRef type. *)
    val affine_map : mltype -> AffineMap.t

    (** Returns the memory space of the given MemRef type. *)
    val memory_space : mltype -> mlattr

    (** Returns the memory spcae of the given Unranked MemRef type. *)
    val unranked_memory_space : mltype -> mlattr
  end

  module Tuple : sig
    (** Checks whether the given type is a tuple type. *)
    val is_tuple : mltype -> bool

    (** Creates a tuple type that consists of the given list of elemental types. The type is owned by the context. *)
    val get : mlcontext -> mltype list -> mltype

    (** Returns the number of types contained in a tuple. *)
    val num_types : mltype -> int

    (** Returns the pos-th type in the tuple type. *)
    val nth : mltype -> int -> mltype
  end

  module Function : sig
    (** Checks whether the given type is a function type. *)
    val is_function : mltype -> bool

    (** Creates a function type, mapping a list of input types to result types. *)
    val get : inputs:mltype list -> results:mltype list -> mlcontext -> mltype

    (** Returns the number of input types. *)
    val num_inputs : mltype -> int

    (** Returns the number of result types. *)
    val num_results : mltype -> int

    (** Returns the pos-th input type. *)
    val input : mltype -> int -> mltype

    (** Returns the pos-th result type. *)
    val result : mltype -> int -> mltype
  end
end

module BuiltinAttributes : sig
  (** Returns an empty attribute. *)
  val null : mlattr

  module AffineMap : sig
    (** Checks whether the given attribute is an affine map attribute. *)
    val is_affine_map : mlattr -> bool

    (** Creates an affine map attribute wrapping the given map. The attribute belongs to the same context as the affine map. *)
    val get : AffineMap.t -> mlattr

    (** Returns the affine map wrapped in the given affine map attribute. *)
    val value : mlattr -> AffineMap.t
  end

  module Array : sig
    (* Checks whether the given attribute is an array attribute. *)
    val is_array : mlattr -> bool

    (** Creates an array element containing the given list of elements in the given context. *)
    val get : mlcontext -> mlattr list -> mlattr

    (** Returns the number of elements stored in the given array attribute. *)
    val num_elements : mlattr -> int

    (** Returns pos-th element stored in the given array attribute. *)
    val element : mlattr -> int -> mlattr
  end

  module Dictionary : sig
    (** Checks whether the given attribute is a dictionary attribute. *)
    val is_dicationary : mlattr -> bool

    (** Creates a dictionary attribute containing the given list of elements in the provided context. *)
    val get : mlcontext -> mlnamed_attr list -> mlattr

    (** Returns the number of attributes contained in a dictionary attribute. *)
    val num_elements : mlattr -> int

    (** Returns pos-th element of the given dictionary attribute. *)
    val element : mlattr -> int -> mlnamed_attr

    (** Returns the dictionary attribute element with the given name or NULL if the given name does not exist in the dictionary. *)
    val element_by_name : mlattr -> string -> mlattr
  end

  module Float : sig
    (* TODO: add support for APFloat and APInt to LLVM IR C API, then expose the relevant functions here. *)

    (** Checks whether the given attribute is a floating point attribute. *)
    val is_float : mlattr -> bool

    (** Creates a floating point attribute in the given context with the given double  and double-precision FP semantics. *)
    val get : mlcontext -> mltype -> float -> mlattr

    (** Same as "mlirFloatAttrDoubleGet", but if the type is not valid for a construction of a FloatAttr, returns a null MlirAttribute. *)
    (* val get_checked : mltype -> float -> mllocation -> mlattr *)

    (** Returns the  stored in the given floating point attribute, interpreting the  as double. *)
    val value : mlattr -> float
  end

  module Integer : sig
    (* TODO: add support for APFloat and APInt to LLVM IR C API, then expose the relevant functions here. *)

    (** Checks whether the given attribute is an integer attribute. *)
    val is_integer : mlattr -> bool

    (** Creates an integer attribute of the given type with the given integer . *)
    val get : mltype -> int -> mlattr

    (** Returns the  stored in the given integer attribute, assuming the value fits into a 64-bit integer. *)
    val value : mlattr -> int
  end

  module Bool : sig
    (** Checks whether the given attribute is a bool attribute. *)
    val is_bool : mlattr -> bool

    (** Creates a bool attribute in the given context with the given . *)
    val get : mlcontext -> int -> mlattr

    (** Returns the  stored in the given bool attribute. *)
    val value : mlattr -> bool
  end

  module IntegerSet : sig
    (** Checks whether the given attribute is an integer set attribute. *)
    val is_integer_set : mlattr -> bool
  end

  module Opaque : sig
    (** Checks whether the given attribute is an opaque attribute. *)
    val is_opaque : mlattr -> bool

    (** Creates an opaque attribute in the given context associated with the dialect identified by its namespace. The attribute contains opaque byte data of the specified length (data need not be null-terminated). *)
    val get : mlcontext -> string -> string -> mltype -> mlattr

    (** Returns the namespace of the dialect with which the given opaque attribute is associated. The namespace string is owned by the context. *)
    val namespace : mlattr -> string

    (** Returns the raw data as a string reference. The data remains live as long as the context in which the attribute lives. *)
    val data : mlattr -> string
  end

  module String : sig
    (** checks whether the given attribute is a string attribute. *)
    val is_string : mlattr -> bool

    (** Creates a string attribute in the given context containing the given string. *)
    val get : mlcontext -> string -> mlattr

    (** Creates a string attribute in the given context containing the given string. Additionally, the attribute has the given type. *)

    val typed_get : mltype -> string -> mlattr

    (** Returns the attribute s as a string reference. The data remains live as long as the context in which the attribute lives. *)
    val value : mlattr -> string
  end

  module SymbolRef : sig
    (** Checks whether the given attribute is a symbol reference attribute. *)
    val is_symbol_ref : mlattr -> bool

    (** Creates a symbol reference attribute in the given context referencing a symbol identified by the given string inside a list of nested references. Each of the references in the list must not be nested. *)
    val get : mlcontext -> string -> mlattr list -> mlattr

    (** Returns the string reference to the root referenced symbol. The data remains live as long as the context in which the attribute lives. *)
    val root_ref : mlattr -> string

    (** Returns the string reference to the leaf referenced symbol. The data remains live as long as the context in which the attribute lives. *)
    val leaf_ref : mlattr -> string

    (** Returns the number of references nested in the given symbol reference attribute. *)
    val num_nested_refs : mlattr -> int

    (** Returns pos-th reference nested in the given symbol reference attribute. *)
    val nested_ref : mlattr -> int -> mlattr
  end

  module FlatSymbolRef : sig
    (** Checks whether the given attribute is a flat symbol reference attribute. *)
    val is_flat_symbol_ref : mlattr -> bool

    (** Creates a flat symbol reference attribute in the given context referencing a symbol identified by the given string. *)
    val get : mlcontext -> string -> mlattr

    (** Returns the referenced symbol as a string reference. The data remains live as long as the context in which the attribute lives. *)
    val value : mlattr -> string
  end

  module Type : sig
    (** Checks whether the given attribute is a type attribute. *)
    val is_type : mlattr -> bool

    (** Creates a type attribute wrapping the given type in the same context as the type. *)
    val get : mltype -> mlattr

    (** Returns the type stored in the given type attribute. *)
    val value : mlattr -> mltype
  end

  module Unit : sig
    (** Checks whether the given attribute is a unit attribute. *)
    val is_unit : mlattr -> bool

    (** Creates a unit attribute in the given context. *)
    val get : mlcontext -> mlattr
  end

  module Elements : sig
    (** Checks whether the given attribute is an elements attribute. *)
    val is_elements : mlattr -> bool

    (** Returns the element at the given rank-dimensional index. *)
    val get : mlattr -> int list -> mlattr

    (** Checks whether the given rank-dimensional index is valid in the given elements attribute. *)
    val is_valid_index : mlattr -> int list -> bool

    (** Gets the total number of elements in the given elements attribute. In order to iterate over the attribute, obtain its type, which must be a statically shaped type and use its sizes to build a multi-dimensional index. *)
    val num_elements : mlattr -> int

    module Dense : sig
      (* TODO: decide on the interface and add support for complex elements. *)
      (* TODO: add support for APFloat and APInt to LLVM IR C API, then expose the relevant functions here. *)

      (** Checks whether the given attribute is a dense elements attribute. *)
      val is_dense : mlattr -> bool

      val is_dense_int : mlattr -> bool
      val is_dense_fpe : mlattr -> bool

      (** Creates a dense elements attribute with the given Shaped type and elements in the same context as the type. *)
      val get : mltype -> mlattr list -> mlattr

      (** Creates a dense elements attribute with the given Shaped type containing a single replicated element (splat). *)
      val splat_get : mltype -> mlattr -> mlattr

      val bool_splat_get : mltype -> bool -> mlattr
      val uint32_splat_get : mltype -> int -> mlattr
      val int32_splat_get : mltype -> int -> mlattr
      val uint64_splat_get : mltype -> int -> mlattr
      val int64_splat_get : mltype -> int -> mlattr
      val float_splat_get : mltype -> float -> mlattr
      val double_splat_get : mltype -> float -> mlattr

      (** Creates a dense elements attribute with the given shaped type from elements of a specific type. Expects the element type of the shaped type to match the * data element type. *)
      val bool_get : mltype -> int list -> mlattr

      val uint32_get : mltype -> int list -> mlattr
      val int32_get : mltype -> int list -> mlattr
      val uint64_get : mltype -> int list -> mlattr
      val int64_get : mltype -> int list -> mlattr
      val float_get : mltype -> float list -> mlattr
      val double_get : mltype -> float list -> mlattr

      (** Creates a dense elements attribute with the given shaped type from string elements. *)
      val string_get : mltype -> string list -> mlattr

      (** Creates a dense elements attribute that has the same data as the given dense elements attribute and a different shaped type. The new type must have the same total number of elements. *)
      val reshape_get : mlattr -> mltype -> mlattr

      (** Checks whether the given dense elements attribute contains a single replicated  (splat). *)
      val is_splat : mlattr -> bool

      (** Returns the single replicated  (splat) of a specific type contained by the given dense elements attribute. *)
      val splat_value : mlattr -> mlattr

      val bool_splat_value : mlattr -> int
      val int32_splat_value : mlattr -> int
      val uint32_splat_value : mlattr -> int
      val int64_splat_value : mlattr -> int
      val uint64_splat_value : mlattr -> int
      val float_splat_value : mlattr -> float
      val double_splat_value : mlattr -> float

      (** Returns the pos-th  (flat contiguous indexing) of a specific type contained by the given dense elements attribute. *)
      val bool_value : mlattr -> int -> bool

      val int32_value : mlattr -> int -> int
      val uint32_value : mlattr -> int -> int
      val int64_value : mlattr -> int -> int
      val uint64_value : mlattr -> int -> int
      val float_value : mlattr -> int -> float
      val double_value : mlattr -> int -> float
      val string_value : mlattr -> int -> string

      (* Returns the raw data of the given dense elements attribute. *)
      (* val raw_data : mlattr -> unit *)
    end

    module Opaque : sig
      (* TODO: expose mldialecto the bindings and implement accessors here. *)

      (** Checks whether the given attribute is an opaque elements attribute. *)
      val is_opaque : mlattr -> bool
    end

    module Sparse : sig
      (** Checks whether the given attribute is a sparse elements attribute. *)
      val is_sparse : mlattr -> bool

      (** Creates a sparse elements attribute of the given shape from a list of indices and a list of associated s. Both lists are expected to be dense elements attributes with the same number of elements. The list of indices is expected to contain 64-bit integers. The attribute is created in the same context as the type. *)
      val create : mltype -> mlattr -> mlattr -> mlattr

      (** Returns the dense elements attribute containing 64-bit integer indices of non-null elements in the given sparse elements attribute. *)
      val indices : mlattr -> mlattr

      (** Returns the dense elements attribute containing the non-null elements in the given sparse elements attribute. *)
      val values : mlattr -> mlattr
    end
  end
end

module TypeIDAllocator : sig
  (* Creates a type id allocator for dynamic type id creation. *)
  val create : unit -> mltypeid_alloc

  (* Deallocates the allocator and all allocated type ids. *)
  val destroy : mltypeid_alloc -> unit

  (* Allocates a type id that is valid for the lifetime of the allocator. *)
  val allocate_type_id : mltypeid_alloc -> mltypeid
end

module PassManager : sig
  (** Create a new top-level PassManager. *)
  val create : mlcontext -> mlpm

  (** Destroy the provided PassManager. *)
  val destroy : mlpm -> unit

  (** Checks if a PassManager is null. *)
  val is_null : mlpm -> bool

  (** Cast a top-level PassManager to a generic OpPassManager. *)
  val to_op_pass_manager : mlpm -> mlop_pm

  (** Run the provided `passManager` on the given `module`. *)
  val run : mlpm -> mlmodule -> bool

  (** Enable mlir-print-ir-after-all. *)
  val enable_ir_printing : mlpm -> unit

  (** Nest an OpPassManager under the top-level PassManager, the nested passmanager will only run on operations matching the provided name. The returned OpPassManager will be destroyed when the parent is destroyed. To further nest more OpPassManager under the newly returned one, see `mlirOpPassManagerNest` below. *)
  val nested_under : mlpm -> string -> mlop_pm

  (** Add a pass and transfer ownership to the provided top-level mlirPassManager. If the pass is not a generic operation pass or a ModulePass, a new OpPassManager is implicitly nested under the provided PassManager. *)
  val add_owned_pass : mlpm -> mlpass -> unit
end

module OpPassManager : sig
  (** Nest an OpPassManager under the provided OpPassManager, the nested passmanager will only run on operations matching the provided name. The returned OpPassManager will be destroyed when the parent is destroyed. *)
  val nested_under : mlop_pm -> string -> mlop_pm

  (** Add a pass and transfer ownership to the provided mlirOpPassManager. If the pass is not a generic operation pass or matching the type of the provided PassManager, a new OpPassManager is implicitly nested under the provided PassManager. *)
  val add_owned_pass : mlop_pm -> mlpass -> unit

  (** Print a textual MLIR pass pipeline by sending chunks of the string representation and forwarding `userData to `callback`. Note that the callback may be called several times with consecutive chunks of the string. *)
  val print_pass_pipeline : callback:(string -> unit) -> mlop_pm -> unit

  (** Parse a textual MLIR pass pipeline and add it to the provided OpPassManager. *)
  (* val parse_pass_pipeline : mlop_pm -> string -> bool *)
end

module ExternalPass : sig
  val empty_callbacks : mlexternal_pass_callbacks

  (* Creates an external `MlirPass` that calls the supplied `callbacks` using the
     supplied `userData`. If `opName` is empty, the pass is a generic operation
     pass. Otherwise it is an operation pass specific to the specified pass name. *)
  val create
    :  mltypeid
    -> name:string
    -> arg:string
    -> desc:string
    -> op_name:string
    -> dep_dialects:mldialect_handle list
    -> mlexternal_pass_callbacks
    -> mlpass

  (* This signals that the pass has failed. This is only valid to call during
     the `run` callback of `MlirExternalPassCallbacks`.
     See Pass::signalPassFailure(). *)
  val signal_failure : mlexternal_pass -> unit
end

module Transforms : sig
  val register_passes : unit -> unit

  module CSE : Transforms_intf.Sig with type t := mlpass
  module Canonicalizer : Transforms_intf.Sig with type t := mlpass
  module ControlFlowSink : Transforms_intf.Sig with type t := mlpass
  module GenerateRuntimeVerification : Transforms_intf.Sig with type t := mlpass
  module Inliner : Transforms_intf.Sig with type t := mlpass
  module LocationSnapshot : Transforms_intf.Sig with type t := mlpass
  module LoopInvariantCodeMotion : Transforms_intf.Sig with type t := mlpass
  module PrintOpStats : Transforms_intf.Sig with type t := mlpass
  module SCCP : Transforms_intf.Sig with type t := mlpass
  module StripDebugInfo : Transforms_intf.Sig with type t := mlpass
  module SymbolDCE : Transforms_intf.Sig with type t := mlpass
  module SymbolPrivatize : Transforms_intf.Sig with type t := mlpass
  module TopologicalSort : Transforms_intf.Sig with type t := mlpass
  module ViewOpGraph : Transforms_intf.Sig with type t := mlpass
end

module RegisterEverything : sig
  (** Appends all upstream dialects and extensions to the dialect registry. *)
  val dialects : mldialect_registry -> unit

  (** Register all translations to LLVM IR for dialects that can support it. *)
  val llvm_translations : mlcontext -> unit

  (** Register all compiler passes of MLIR. *)
  val passes : unit -> unit
end

(** [with_context f]  creates a context [ctx], applies [f] to it, destroys it and returns the result of applying [f] *)
val with_context : (mlcontext -> 'a) -> 'a

(** [with_pass_manager f ctx]  creates a Pass Manager [pm] for the given context [ctx], applies [f] to it, destroys it and returns the result of applying [f] *)
val with_pass_manager : f:(mlpm -> 'a) -> mlcontext -> 'a
