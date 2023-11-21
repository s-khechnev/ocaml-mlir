open Ctypes
open Wrapper

type 'a structured = ('a, [ `Struct ]) Ctypes_static.structured
type mlcontext = Typs.Context.t structured
type mldialect = Typs.Dialect.t structured
type mltype = Typs.Type.t structured
type mltypeid = Typs.TypeID.t structured
type mlblock = Typs.Block.t structured
type mlregion = Typs.Region.t structured
type mlvalue = Typs.Value.t structured
type mllocation = Typs.Location.t structured
type mlmodule = Typs.Module.t structured
type mlop = Typs.Operation.t structured
type mlop_state = Typs.OperationState.t structured
type mlattr = Typs.Attribute.t structured
type mlnamed_attr = Typs.NamedAttribute.t structured
type mlpass = Typs.Pass.t structured
type mlpm = Typs.PassManager.t structured
type mlop_pm = Typs.OpPassManager.t structured
type mlident = Typs.Identifier.t structured
type mlaffine_expr = Typs.AffineExpr.t structured
type mldialect_handle = Typs.DialectHandle.t structured
type mlsymboltbl = Typs.SymbolTable.t structured
type mlop_operand = Typs.OpOperand.t structured
type mltypeid_alloc = Typs.TypeIDAllocator.t structured
type mlexternal_pass = Typs.ExternalPass.t structured

type mlexternal_pass_callbacks =
  { construct : unit -> unit
  ; destruct : unit -> unit
  ; initialize : (mlcontext -> unit -> bool) option
  ; clone : unit -> unit
  ; run : mlop -> mlexternal_pass -> unit
  }

module StringRef = Bindings.StringRef

module IR = struct
  module Context = struct
    include Bindings.Context

    let global_ctx = create ()
    let num_registered_dialects ctx = num_registered_dialects ctx |> Signed.Long.to_int
    let num_loaded_dialects ctx = num_loaded_dialects ctx |> Signed.Long.to_int
    let get_or_load_dialect ctx s = get_or_load_dialect ctx StringRef.(of_string s)
  end

  module Dialect = struct
    include Bindings.Dialect

    let namespace dialect = namespace dialect |> StringRef.to_string
  end

  module DialectHandle = struct
    include Bindings.DialectHandle

    let register dhandle ctx = register_dialect dhandle ctx
  end

  module Type = struct
    include Bindings.Type

    let parse ctx s =
      let s = StringRef.of_string s in
      parse ctx s


    let print ~callback x =
      let callback s _ = callback (StringRef.to_string s) in
      print x callback null
  end

  module Location = struct
    include Bindings.Location

    let file_line_col_get ctx s i j =
      file_line_col_get
        ctx
        Bindings.StringRef.(of_string s)
        Unsigned.UInt.(of_int i)
        Unsigned.UInt.(of_int j)


    let unknown = unknown

    let print ~callback x =
      let callback s _ = callback (StringRef.to_string s) in
      print x callback null
  end

  module Attribute = struct
    include Bindings.Attribute

    let parse ctx s =
      let s = StringRef.of_string s in
      parse ctx s


    let print ~callback x =
      let callback s _ = callback (StringRef.to_string s) in
      print x callback null


    let name id attr = name id attr
  end

  module OperationState = struct
    include Bindings.OperationState

    let get s loc =
      let s = StringRef.of_string s in
      get s loc


    let add_results opstate results =
      let opstate = addr opstate in
      let n = List.length results |> Intptr.of_int in
      let results = CArray.(start (of_list Typs.Type.t results)) in
      add_results opstate n results


    let add_named_attributes opstate attrs =
      let opstate = addr opstate in
      let n = List.length attrs |> Intptr.of_int in
      let attrs = CArray.(start (of_list Typs.NamedAttribute.t attrs)) in
      add_attributes opstate n attrs


    let add_owned_regions opstate regions =
      let opstate = addr opstate in
      let n = List.length regions |> Intptr.of_int in
      let regions = CArray.(start (of_list Typs.Region.t regions)) in
      add_owned_regions opstate n regions


    let add_operands opstate operands =
      let opstate = addr opstate in
      let n = List.length operands |> Intptr.of_int in
      let operands = CArray.(start (of_list Typs.Value.t operands)) in
      add_operands opstate n operands
  end

  module Operation = struct
    include Bindings.Operation

    let create opstate =
      let opstate = addr opstate in
      Bindings.Operation.create opstate


    let name op = name op |> Bindings.Identifier.to_string |> StringRef.to_string
    let region x pos = Bindings.Operation.region x Intptr.(of_int pos)
    let num_regions reg = num_regions reg |> Intptr.to_int
    let num_operands op = num_operands op |> Intptr.to_int
    let operand x pos = Bindings.Operation.operand x Intptr.(of_int pos)
    let set_operand op pos value = set_operand op Intptr.(of_int pos) value
    let num_results op = num_results op |> Intptr.to_int
    let result x pos = Bindings.Operation.result x Intptr.(of_int pos)
    let attribute_by_name op name = attribute_by_name op (StringRef.of_string name)

    let set_attribute_by_name op name attr =
      set_attribute_by_name op (StringRef.of_string name) attr
  end

  module Value = struct
    include Bindings.Value

    let block_argument_arg_num x = block_argument_arg_num x |> Intptr.to_int
    let op_result_get_result_num x = op_result_get_result_num x |> Intptr.to_int

    let print ~callback x =
      let callback s _ = callback (StringRef.to_string s) in
      print x callback null
  end

  module OpOperand = struct
    include Bindings.OpOperand

    let operand_number oper = operand_number oper |> Unsigned.UInt.to_int
  end

  module Block = struct
    include Bindings.Block

    let create typs loc =
      let size = List.length typs |> Intptr.of_int in
      let typs = CArray.(start (of_list Typs.Type.t typs)) in
      Bindings.Block.create size typs (Ctypes.allocate Typs.Location.t loc)


    let insert_owned_operation blk pos f =
      let pos = Intptr.of_int pos in
      Bindings.Block.insert_owned_operation blk pos f


    let num_arguments blk = Bindings.Block.num_arguments blk |> Intptr.to_int

    let argument x pos =
      let pos = Intptr.of_int pos in
      Bindings.Block.argument x pos


    let ops blk =
      let rec loop op acc =
        if Operation.is_null op then acc else loop (Operation.next_in_block op) (op :: acc)
      in
      loop (first_operation blk) [] |> List.rev
  end

  module Module = struct
    include Bindings.Module

    let parse ctx str = parse ctx StringRef.(of_string str)
  end

  module Region = struct
    include Bindings.Region
  end

  module Identifier = struct
    include Bindings.Identifier

    let get ctx str = get ctx (StringRef.of_string str)
    let to_string id = to_string id |> StringRef.to_string
  end

  module SymbolTable = struct
    include Bindings.SymbolTable

    let symbol_attr_name () = symbol_attr_name () |> StringRef.to_string
    let visibility_attr_name () = visibility_attr_name () |> StringRef.to_string
    let lookup tbl name = lookup tbl (StringRef.of_string name)

    let replace_all_symbol_uses ~old_sym ~new_sym from =
      replace_all_symbol_uses
        (StringRef.of_string old_sym)
        (StringRef.of_string new_sym)
        from
      |> Bindings.LogicalResult.is_success
  end
end

module TypeIDAllocator = struct
  include Bindings.TypeIDAllocator
end

module AffineExpr = struct
  type t = Typs.AffineExpr.t structured

  include Bindings.AffineExpr

  let print ~callback x =
    let callback s _ = callback (StringRef.to_string s) in
    print x callback null


  let largest_known_divisor x = largest_known_divisor x |> Int64.to_int
  let is_multiple_of x i = is_multiple_of x Int64.(of_int i)
  let is_function_of_dim x i = is_function_of_dim x Intptr.(of_int i)

  module Dimension = struct
    include Bindings.AffineExpr.Dimension

    let get ctx i = get ctx Intptr.(of_int i)
    let position x = position x |> Intptr.to_int
  end

  module Symbol = struct
    include Bindings.AffineExpr.Symbol

    let get ctx i = get ctx Intptr.(of_int i)
    let position x = position x |> Intptr.to_int
  end

  module Constant = struct
    include Bindings.AffineExpr.Constant

    let get ctx i = get ctx Int64.(of_int i)
    let value x = value x |> Int64.to_int
  end
end

module AffineMap = struct
  type t = Typs.AffineMap.t structured

  include Bindings.AffineMap

  let get ctx i j k expr =
    get
      ctx
      Intptr.(of_int i)
      Intptr.(of_int j)
      Intptr.(of_int k)
      (Ctypes.allocate Typs.AffineExpr.t expr)


  let constant ctx i = constant ctx Int64.(of_int i)

  let permutation ctx perm =
    let size = List.length perm |> Intptr.of_int in
    let perm =
      let perm = List.map Unsigned.UInt.of_int perm in
      CArray.(start (of_list uint perm))
    in
    permutation ctx size perm


  let multi_dim_identity ctx i = multi_dim_identity ctx Intptr.(of_int i)
  let minor_identity ctx i j = minor_identity ctx Intptr.(of_int i) (Intptr.of_int j)
  let single_constant_result ctx = single_constant_result ctx |> Int64.to_int
  let num_dims ctx = num_dims ctx |> Intptr.to_int
  let num_symbols ctx = num_symbols ctx |> Intptr.to_int
  let num_results ctx = num_results ctx |> Intptr.to_int
  let num_inputs ctx = num_inputs ctx |> Intptr.to_int

  let sub_map afm x =
    let size = List.length x |> Intptr.of_int in
    let x =
      let x = List.map Intptr.of_int x in
      CArray.(start (of_list intptr_t x))
    in
    sub_map afm size x


  let major_sub_map ctx i = major_sub_map ctx (Intptr.of_int i)
  let minor_sub_map ctx i = minor_sub_map ctx (Intptr.of_int i)

  let print ~callback x =
    let callback s _ = callback (StringRef.to_string s) in
    print x callback null
end

module PassManager = struct
  include Bindings.PassManager

  let run pass m = Bindings.LogicalResult.(is_success (run pass m))
  let nested_under pm s = nested_under pm StringRef.(of_string s)
end

module OpPassManager = struct
  include Bindings.OpPassManager

  let nested_under pm s = nested_under pm StringRef.(of_string s)

  let print_pass_pipeline ~callback x =
    let callback s _ = callback (StringRef.to_string s) in
    print_pass_pipeline x callback null


  let parse_pass_pipeline pm s ~callback =
    let callback s _ = callback (StringRef.to_string s) in
    parse_pass_pipeline pm StringRef.(of_string s) callback null
    |> Bindings.LogicalResult.is_success
end

module ExternalPass = struct
  include Bindings.ExternalPass

  let empty_callbacks =
    { construct = (fun _ -> ())
    ; destruct = (fun _ -> ())
    ; initialize = None
    ; clone = (fun _ -> ())
    ; run = (fun _ _ -> ())
    }


  let create type_id ~name ~arg ~desc ~op_name ~dep_dialects callbacks =
    let s_callbacks = make ExternalPassCallbacks.t in
    let () =
      setf s_callbacks ExternalPassCallbacks.construct (fun _ -> callbacks.construct ());
      setf s_callbacks ExternalPassCallbacks.destruct (fun _ -> callbacks.destruct ());
      (match callbacks.initialize with
       | Some init ->
         setf s_callbacks ExternalPassCallbacks.initialize (fun ctx _ ->
           if init ctx ()
           then Bindings.LogicalResult.success ()
           else Bindings.LogicalResult.failure ())
       | None -> ());
      setf s_callbacks ExternalPassCallbacks.clone (fun _ ->
        let _ = callbacks.clone () in
        null);
      setf s_callbacks ExternalPassCallbacks.run (fun op pass _ -> callbacks.run op pass)
    in
    create
      type_id
      (StringRef.of_string name)
      (StringRef.of_string arg)
      (StringRef.of_string desc)
      (StringRef.of_string op_name)
      (Intptr.of_int (List.length dep_dialects))
      CArray.(start (of_list Typs.DialectHandle.t dep_dialects))
      s_callbacks
      null
end

module BuiltinTypes = struct
  module Integer = struct
    include Bindings.BuiltinTypes.Integer

    let get ctx i = get ctx Unsigned.UInt.(of_int i)
    let signed ctx i = signed ctx Unsigned.UInt.(of_int i)
    let unsigned ctx i = unsigned ctx Unsigned.UInt.(of_int i)
    let width typ = width typ |> Unsigned.UInt.to_int
  end

  module Float = Bindings.BuiltinTypes.Float
  module Index = Bindings.BuiltinTypes.Index
  module None = Bindings.BuiltinTypes.None
  module Complex = Bindings.BuiltinTypes.Complex
  module Shaped = Bindings.BuiltinTypes.Shaped

  module Vector = struct
    include Bindings.BuiltinTypes.Vector

    let get shp typ =
      let n = Array.length shp in
      let shp =
        let shp = shp |> Array.map Int64.of_int |> Array.to_list in
        CArray.(start (of_list int64_t shp))
      in
      get Intptr.(of_int n) shp typ


    let get_checked loc rank shp typ =
      let shp =
        let shp = shp |> Array.map Int64.of_int |> Array.to_list in
        CArray.(start (of_list int64_t shp))
      in
      get_checked loc Intptr.(of_int rank) shp typ
  end

  module Tensor = struct
    include Bindings.BuiltinTypes.Tensor

    let ranked shp typ attr =
      let rank = Array.length shp in
      let shp =
        let shp = shp |> Array.map Int64.of_int |> Array.to_list in
        CArray.(start (of_list int64_t shp))
      in
      ranked Intptr.(of_int rank) shp typ attr


    let ranked_checked loc shp typ encoding =
      let rank = Array.length shp in
      let shp =
        let shp = shp |> Array.map Int64.of_int |> Array.to_list in
        CArray.(start (of_list int64_t shp))
      in
      ranked_checked loc Intptr.(of_int rank) shp typ encoding
  end

  module MemRef = struct
    include Bindings.BuiltinTypes.MemRef

    let get typ rank shape layout memspace =
      let shape =
        let shp = shape |> Array.map Int64.of_int |> Array.to_list in
        CArray.(start (of_list int64_t shp))
      in
      get typ Intptr.(of_int rank) shape layout memspace


    let contiguous typ rank shape memspace =
      let shape =
        let shp = shape |> Array.map Int64.of_int |> Array.to_list in
        CArray.(start (of_list int64_t shp))
      in
      contiguous typ Intptr.(of_int rank) shape memspace


    let contiguous_checked typ rank shape memspace loc =
      let shape =
        let shp = shape |> Array.map Int64.of_int |> Array.to_list in
        CArray.(start (of_list int64_t shp))
      in
      contiguous_checked loc typ Intptr.(of_int rank) shape memspace


    let unranked typ memspace = unranked typ memspace
    let unranked_checked loc typ memspace = unranked_checked loc typ memspace
    (* let num_affine_maps typ = num_affine_maps typ |> Intptr.to_int
       let affine_map typ i = affine_map typ Intptr.(of_int i)
       let memory_space typ = memory_space typ |> Unsigned.UInt.to_int
       let unranked_memory_space typ = unranked_memory_space typ |> Unsigned.UInt.to_int *)
  end

  module Tuple = struct
    include Bindings.BuiltinTypes.Tuple

    let get ctx typs =
      let n = List.length typs |> Intptr.of_int in
      let typs = CArray.(start (of_list Typs.Type.t typs)) in
      get ctx n typs


    let num_types typ = num_types typ |> Intptr.to_int
    let nth typ pos = get_type typ Intptr.(of_int pos)
  end

  module Function = struct
    include Bindings.BuiltinTypes.Function

    let get ~inputs ~results ctx =
      let n_inputs = List.length inputs |> Intptr.of_int in
      let inputs = CArray.(start (of_list Typs.Type.t inputs)) in
      let n_results = List.length results |> Intptr.of_int in
      let results = CArray.(start (of_list Typs.Type.t results)) in
      get ctx n_inputs inputs n_results results


    let num_inputs typ = num_inputs typ |> Intptr.to_int
    let num_results typ = num_results typ |> Intptr.to_int
    let input typ i = input typ (Intptr.of_int i)
    let result typ i = result typ (Intptr.of_int i)
  end
end

module BuiltinAttributes = struct
  let null = Bindings.BuiltinAttributes.null ()

  module AffineMap = Bindings.BuiltinAttributes.AffineMap

  module Array = struct
    include Bindings.BuiltinAttributes.Array

    let get ctx x =
      let size = List.length x |> Intptr.of_int in
      let x = CArray.(start (of_list Typs.Attribute.t x)) in
      get ctx size x


    let num_elements x = num_elements x |> Intptr.to_int
    let element x pos = element x Intptr.(of_int pos)
  end

  module Dictionary = struct
    include Bindings.BuiltinAttributes.Dictionary

    let get ctx x =
      let size = List.length x |> Intptr.of_int in
      let x = CArray.(start (of_list Typs.NamedAttribute.t x)) in
      get ctx size x


    let num_elements x = num_elements x |> Intptr.to_int
    let element x pos = element x Intptr.(of_int pos)
    let element_by_name x key = element_by_name x StringRef.(of_string key)
  end

  module Float = Bindings.BuiltinAttributes.Float

  module Integer = struct
    include Bindings.BuiltinAttributes.Integer

    let get x i = get x Int64.(of_int i)
    let value x = value_int x |> Int64.to_int
  end

  module Bool = Bindings.BuiltinAttributes.Bool
  module IntegerSet = Bindings.BuiltinAttributes.IntegerSet

  module Opaque = struct
    include Bindings.BuiltinAttributes.Opaque

    let get ctx namespace s typs =
      let len = String.length s in
      let s = CArray.of_string s |> CArray.start in
      get ctx StringRef.(of_string namespace) Intptr.(of_int len) s typs


    let namespace x = namespace x |> StringRef.to_string
    let data x = data x |> StringRef.to_string
  end

  module String = struct
    include Bindings.BuiltinAttributes.String

    let get ctx s = get ctx StringRef.(of_string s)
    let typed_get typ s = typed_get typ StringRef.(of_string s)
    let value s = value s |> StringRef.to_string
  end

  module SymbolRef = struct
    include Bindings.BuiltinAttributes.SymbolRef

    let get ctx s attrs =
      let size = List.length attrs |> Intptr.of_int in
      let attrs = CArray.(start (of_list Typs.Attribute.t attrs)) in
      get ctx StringRef.(of_string s) size attrs


    let root_ref attr = root_ref attr |> StringRef.to_string
    let leaf_ref attr = leaf_ref attr |> StringRef.to_string
    let num_nested_refs x = num_nested_refs x |> Intptr.to_int
    let nested_ref x i = nested_ref x Intptr.(of_int i)
  end

  module FlatSymbolRef = struct
    include Bindings.BuiltinAttributes.FlatSymbolRef

    let get ctx s = get ctx StringRef.(of_string s)
    let value s = value s |> StringRef.to_string
  end

  module Type = Bindings.BuiltinAttributes.Type
  module Unit = Bindings.BuiltinAttributes.Unit

  module Elements = struct
    include Bindings.BuiltinAttributes.Elements

    let get attr xs =
      let size = List.length xs |> Intptr.of_int in
      let xs =
        let xs = List.map Unsigned.UInt64.of_int xs in
        CArray.(start (of_list uint64_t xs))
      in
      get attr size xs


    let is_valid_index attr xs =
      let size = List.length xs |> Intptr.of_int in
      let xs =
        let xs = List.map Unsigned.UInt64.of_int xs in
        CArray.(start (of_list uint64_t xs))
      in
      is_valid_index attr size xs


    let num_elements attrs = num_elements attrs |> Intptr.to_int

    module Sparse = Bindings.BuiltinAttributes.Sparse
    module Opaque = Bindings.BuiltinAttributes.Opaque

    module Dense = struct
      include Bindings.BuiltinAttributes.Dense

      let get attr xs =
        let size = List.length xs |> Intptr.of_int in
        let xs = CArray.(start (of_list Typs.Attribute.t xs)) in
        get attr size xs


      let uint32_splat_get typ i = uint32_splat_get typ Unsigned.UInt32.(of_int i)
      let int32_splat_get typ i = int32_splat_get typ Int32.(of_int i)
      let uint64_splat_get typ i = uint64_splat_get typ Unsigned.UInt64.(of_int i)
      let int64_splat_get typ i = int64_splat_get typ Int64.(of_int i)

      let _wrapper_get f t g =
        let dummy typ xs =
          let xs = List.map g xs in
          let size = List.length xs |> Intptr.of_int in
          let xs = CArray.(start (of_list t xs)) in
          f typ size xs
        in
        dummy


      let bool_get = _wrapper_get bool_get int (fun x -> x)
      let uint32_get = _wrapper_get uint32_get uint32_t Unsigned.UInt32.of_int
      let int32_get = _wrapper_get int32_get int32_t Int32.of_int
      let uint64_get = _wrapper_get uint64_get uint64_t Unsigned.UInt64.of_int
      let int64_get = _wrapper_get int64_get int64_t Int64.of_int
      let float_get = _wrapper_get float_get float (fun x -> x)
      let double_get = _wrapper_get double_get double (fun x -> x)
      let string_get = _wrapper_get string_get Typs.StringRef.t StringRef.of_string
      let int32_splat_value x = int32_splat_value x |> Int32.to_int
      let uint32_splat_value x = uint32_splat_value x |> Unsigned.UInt32.to_int
      let int64_splat_value x = int64_splat_value x |> Int64.to_int
      let uint64_splat_value x = uint64_splat_value x |> Unsigned.UInt64.to_int
      let bool_value x i = bool_value x Intptr.(of_int i)
      let int32_value x i = int32_value x Intptr.(of_int i) |> Int32.to_int
      let uint32_value x i = uint32_value x Intptr.(of_int i) |> Unsigned.UInt32.to_int
      let int64_value x i = int64_value x Intptr.(of_int i) |> Int64.to_int
      let uint64_value x i = uint64_value x Intptr.(of_int i) |> Unsigned.UInt64.to_int
      let float_value x i = float_value x Intptr.(of_int i)
      let double_value x i = double_value x Intptr.(of_int i)
      let string_value x i = string_value x Intptr.(of_int i) |> StringRef.to_string
    end
  end
end

module Transforms = Bindings.Transforms

let register_all_dialects = Bindings.register_all_dialects

let with_context f =
  let ctx = IR.Context.create () in
  let result = f ctx in
  IR.Context.destroy ctx;
  result


let with_pass_manager ~f ctx =
  let pm = PassManager.create ctx in
  let result = f pm in
  PassManager.destroy pm;
  result
