open Base
open Mlir
open IR

let f64 = BuiltinTypes.Float.f64 Context.global_ctx

let memref_of_tensor tensor =
  let () = assert (BuiltinTypes.Tensor.is_ranked_tensor tensor) in
  let rank = BuiltinTypes.Shaped.rank tensor in
  let shape = Array.init rank ~f:(BuiltinTypes.Shaped.dim_size tensor) in
  BuiltinTypes.MemRef.get f64 shape BuiltinAttributes.null BuiltinAttributes.null


let alloc_dealloc typ loc blk =
  let alloc =
    let alloc_st = OperationState.get "memref.alloc" loc in
    let () =
      OperationState.add_named_attributes
        alloc_st
        [ Attribute.name
            "operand_segment_sizes"
            (BuiltinAttributes.Dense.Array.i32 Context.global_ctx [ 0; 0 ])
        ]
    in
    let () = OperationState.add_results alloc_st [ typ ] in
    Operation.create alloc_st
  in
  let () =
    let fst_op = Block.first_operation blk in
    Block.insert_owned_operation_before blk fst_op alloc
  in
  let alloc_val = Operation.result alloc 0 in
  let dealloc =
    let dealloc_st = OperationState.get "memref.dealloc" loc in
    let () = OperationState.add_operands dealloc_st [ alloc_val ] in
    Operation.create dealloc_st
  in
  let () =
    let term_op = Block.terminator blk in
    Block.insert_owned_operation_before blk term_op dealloc
  in
  alloc_val


let map mref_typ =
  let rank = BuiltinTypes.Shaped.rank mref_typ in
  if rank = 0
  then AffineMap.empty Context.global_ctx
  else AffineMap.multi_dim_identity Context.global_ctx rank


let affine_store_op loc value_to_store memref indices =
  let mref_typ = Value.get_type memref in
  let op_st = OperationState.get "affine.store" loc in
  OperationState.add_operands op_st ([ value_to_store; memref ] @ indices);
  OperationState.add_named_attributes
    op_st
    [ Attribute.name "map" (BuiltinAttributes.AffineMap.get (map mref_typ)) ];
  Operation.create op_st


let affine_load_op loc indices alloc =
  let op_st = OperationState.get "affine.load" loc in
  let mref_typ = Value.get_type alloc in
  OperationState.add_named_attributes
    op_st
    [ Attribute.name "map" (BuiltinAttributes.AffineMap.get (map mref_typ)) ];
  OperationState.add_operands op_st [ alloc ];
  OperationState.add_operands op_st indices;
  OperationState.add_results op_st [ f64 ];
  Operation.create op_st


let build_loop_from_consts loc lb ub body_builder =
  let for_op_st = OperationState.get "affine.for" loc in
  let () =
    let lb_map = AffineMap.constant Context.global_ctx lb in
    let ub_map = AffineMap.constant Context.global_ctx ub in
    let step_attr =
      let step_val = 1 in
      let step =
        BuiltinAttributes.Integer.get (BuiltinTypes.Index.get Context.global_ctx) step_val
      in
      Attribute.name "step" step
    in
    let bound_attr name v = Attribute.name name (BuiltinAttributes.AffineMap.get v) in
    OperationState.add_named_attributes
      for_op_st
      [ step_attr; bound_attr "lower_bound" lb_map; bound_attr "upper_bound" ub_map ]
  in
  let entry_block = Block.create [ BuiltinTypes.Index.get Context.global_ctx ] [ loc ] in
  let induction_var = Block.argument entry_block 0 in
  let () = body_builder loc entry_block induction_var in
  let () =
    let region = Region.create () in
    let () = Region.append_owned_block region entry_block in
    OperationState.add_owned_regions for_op_st [ region ]
  in
  Operation.create for_op_st


let build_loop_nest loc lower_bounds upper_bounds body_builder =
  let () = assert (Array.(length lower_bounds = length upper_bounds)) in
  let count = Array.length lower_bounds in
  if count = 0
  then (
    (* If there are no loops to be constructed, construct the body anyway. *)
    let dummy_blk = Block.create [] [ loc ] in
    body_builder dummy_blk loc [];
    Block.ops dummy_blk)
  else (
    let ivs = Queue.create () in
    let rec loop i ?parent_blk ops =
      if i = count
      then (
        List.iter ops ~f:(fun op ->
          let op_blk = Region.first_block @@ Operation.region op 0 in
          let yield_op = Operation.create @@ OperationState.get "affine.yield" loc in
          Block.append_owned_operation op_blk yield_op);
        [ List.last_exn ops ])
      else (
        let loop_body loc blk iv =
          let () = Queue.enqueue ivs iv in
          if i = count - 1 (* reached innermost *)
          then body_builder blk loc (Queue.to_list ivs)
        in
        let for_op =
          build_loop_from_consts loc lower_bounds.(i) upper_bounds.(i) loop_body
        in
        let () =
          match parent_blk with
          | Some blk -> Block.append_owned_operation blk for_op
          | None -> ()
        in
        let for_blk = Region.first_block @@ Operation.region for_op 0 in
        loop (i + 1) ~parent_blk:for_blk (for_op :: ops))
    in
    loop 0 [])


let lower_op_to_loops op blk body_builder =
  let tensor_typ = Value.get_type @@ Operation.result op 0 in
  let loc = Operation.loc op in
  (* Insert an allocation and deallocation for the result of this operation. *)
  let mref_typ = memref_of_tensor tensor_typ in
  let alloc = alloc_dealloc mref_typ loc blk in
  (* Create a nest of affine loops, with one loop per dimension of the shape.
     The `build_loop_nest` function takes a callback that is used to construct
     the body of the innermost loop given a builder, a location and a range of
     loop induction variables. *)
  let rank = BuiltinTypes.Shaped.rank tensor_typ in
  let lower_bounds = Array.init rank ~f:(fun _ -> 0) in
  let shape = Array.init rank ~f:(BuiltinTypes.Shaped.dim_size tensor_typ) in
  let body_builder blk loc ivs =
    let value_to_store = body_builder blk loc ivs in
    let op = Value.op_result_get_owner value_to_store in
    let blk = Operation.block op in
    let affine_store = affine_store_op loc value_to_store alloc ivs in
    Block.insert_owned_operation_after blk op affine_store
  in
  Value.replace_uses ~old:(Operation.result op 0) ~fresh:alloc;
  let ops = build_loop_nest loc lower_bounds shape body_builder in
  Block.insert_ops_after blk op ops


let lower_func op =
  if Operation.(num_operands op <> 0 || num_results op <> 0)
  then failwith "expected 'main' to have 0 inputs and 0 results";
  let func_op =
    let op = Operation.clone op in
    let func_op_st = OperationState.get "func.func" (Operation.loc op) in
    let attrs = List.init (Operation.num_attributes op) ~f:(Operation.attribute op) in
    (* for jit *)
    let emit_c_interface_attr =
      Attribute.name
        "llvm.emit_c_interface"
        (BuiltinAttributes.Unit.get Context.global_ctx)
    in
    OperationState.add_named_attributes func_op_st (emit_c_interface_attr :: attrs);
    let reg = Region.create () in
    let main_blk = Region.first_block (Operation.region op 0) in
    Region.append_owned_block reg main_blk;
    OperationState.add_owned_regions func_op_st [ reg ];
    Operation.create func_op_st
  in
  let modul_blk = Operation.block op in
  Block.append_owned_operation modul_blk func_op;
  Operation.destroy op


let lower_op op blk =
  match Operation.name op with
  | "toy.constant" ->
    (* When lowering the constant operation, we allocate and assign the constant
       values to a corresponding memref allocation. *)
    let loc = Operation.loc op in
    let tensor_typ = Value.get_type @@ Operation.result op 0 in
    let mref_typ = memref_of_tensor tensor_typ in
    let alloc = alloc_dealloc mref_typ loc blk in
    (* We will be generating constant indices up-to the largest dimension.
       Create these constants up-front to avoid large amounts of redundant
       operations. *)
    let value_shape =
      List.init
        (BuiltinTypes.Shaped.rank mref_typ)
        ~f:(BuiltinTypes.Shaped.dim_size mref_typ)
    in
    let const_indices =
      let build_const_idx idx =
        let const_id_st = OperationState.get "arith.constant" loc in
        let () =
          let idx =
            BuiltinAttributes.Integer.get (BuiltinTypes.Index.get Context.global_ctx) idx
          in
          OperationState.add_named_attributes const_id_st [ Attribute.name "value" idx ]
        in
        let () =
          OperationState.add_results
            const_id_st
            [ BuiltinTypes.Index.get Context.global_ctx ]
        in
        Operation.create const_id_st
      in
      match List.max_elt value_shape ~compare:Int.compare with
      | Some max -> Array.init max ~f:build_const_idx
      | None -> [| build_const_idx 0 |]
    in
    (* The constant operation represents a multi-dimensional constant, so we
       will need to generate a store for each of the elements. *)
    let ops =
      (* Generate all coordinates of tensor. 
          e.g for [[1; 2]; [3; 4]] -> [[0; 0]; [0; 1]; [1; 0]; [1; 1] *)
      let coords =
        let rec gen_coords shape acc =
          match shape with
          | [] -> [ List.rev acc ]
          | dim :: rest ->
            List.concat_map (List.init dim ~f:Fn.id) ~f:(fun c ->
              gen_coords rest (c :: acc))
        in
        gen_coords value_shape []
      in
      List.fold coords ~init:[] ~f:(fun acc coord ->
        let const_op =
          let const_op_st = OperationState.get "arith.constant" loc in
          let value =
            let value_attr = Operation.attribute_by_name op "value" in
            let value =
              BuiltinAttributes.Float.value
                (BuiltinAttributes.Elements.get value_attr coord)
            in
            BuiltinAttributes.Float.get Context.global_ctx f64 value
          in
          OperationState.add_named_attributes const_op_st [ Attribute.name "value" value ];
          OperationState.add_results const_op_st [ f64 ];
          Operation.create const_op_st
        in
        let indices = List.map coord ~f:(fun i -> Operation.result const_indices.(i) 0) in
        let affine_store_op =
          affine_store_op loc (Operation.result const_op 0) alloc indices
        in
        [ const_op; affine_store_op ] @ acc)
    in
    Value.replace_uses ~old:(Operation.result op 0) ~fresh:alloc;
    Block.insert_ops_after blk op (Array.to_list const_indices @ ops)
  | "toy.add" | "toy.mul" ->
    let body_builder blk loc ivs =
      let lhs, rhs = Operation.(operand op 0, operand op 1) in
      let loaded_lhs, loaded_rhs =
        let affine_load_op = affine_load_op loc ivs in
        affine_load_op lhs, affine_load_op rhs
      in
      let lowered_bin_op =
        let bin_op name =
          let op_st = OperationState.get name loc in
          OperationState.add_operands
            op_st
            Operation.[ result loaded_lhs 0; result loaded_rhs 0 ];
          OperationState.add_results op_st [ f64 ];
          Operation.create op_st
        in
        match Operation.name op with
        | "toy.add" -> bin_op "arith.addf"
        | "toy.mul" -> bin_op "arith.mulf"
        | _ -> assert false
      in
      Block.append_owned_operation blk loaded_lhs;
      Block.append_owned_operation blk loaded_rhs;
      Block.append_owned_operation blk lowered_bin_op;
      Operation.result lowered_bin_op 0
    in
    lower_op_to_loops op blk body_builder
  | "toy.return" ->
    if Operation.num_operands op <> 0 then failwith "Return must not have an operand";
    let func_ret_op =
      Operation.create @@ OperationState.get "func.return" (Operation.loc op)
    in
    Block.insert_owned_operation_before blk op func_ret_op
  | "toy.transpose" ->
    let body_builder blk loc ivs =
      let input = Operation.operand op 0 in
      let ivs = List.rev ivs in
      let load = affine_load_op loc ivs input in
      Block.append_owned_operation blk load;
      Operation.result load 0
    in
    lower_op_to_loops op blk body_builder
  | "toy.print" ->
    (* it's op casts ranked memref to unranked *)
    let cast_op =
      let cast_op_st = OperationState.get "memref.cast" (Operation.loc op) in
      OperationState.add_operands cast_op_st [ Operation.operand op 0 ];
      OperationState.add_results
        cast_op_st
        [ BuiltinTypes.MemRef.unranked f64 BuiltinAttributes.null ];
      Operation.create cast_op_st
    in
    (* get or insert forward-declaration of "print" *)
    let get_or_insert_print () =
      let print_func_name = "printMemrefF64" in
      let () =
        let modul_blk =
          let modul = Operation.(parent @@ parent op) in
          Module.(body @@ from_op modul)
        in
        (* if print declaration not found, then create it *)
        if not
           @@ List.exists (Block.ops modul_blk) ~f:(fun op ->
             let func_name =
               BuiltinAttributes.String.value (Operation.attribute_by_name op "sym_name")
             in
             String.equal func_name print_func_name)
        then (
          let func_print_op =
            let func_op_st = OperationState.get "func.func" (Operation.loc op) in
            let () =
              (* for jit *)
              let emit_c_interface_attr =
                Attribute.name
                  "llvm.emit_c_interface"
                  (BuiltinAttributes.Unit.get Context.global_ctx)
              in
              let name_attr =
                Attribute.name
                  "sym_name"
                  (BuiltinAttributes.String.get Context.global_ctx print_func_name)
              in
              let visibility_attr =
                Attribute.name
                  "sym_visibility"
                  (BuiltinAttributes.String.get Context.global_ctx "private")
              in
              let func_typ_attr =
                let func_typ =
                  BuiltinTypes.Function.get
                    Context.global_ctx
                    ~inputs:[ BuiltinTypes.MemRef.unranked f64 BuiltinAttributes.null ]
                    ~results:[]
                in
                Attribute.name "function_type" (BuiltinAttributes.Type.get func_typ)
              in
              OperationState.add_named_attributes
                func_op_st
                [ emit_c_interface_attr; name_attr; visibility_attr; func_typ_attr ]
            in
            OperationState.add_owned_regions func_op_st [ Region.create () ];
            Operation.create func_op_st
          in
          Block.append_owned_operation modul_blk func_print_op)
      in
      BuiltinAttributes.FlatSymbolRef.get Context.global_ctx print_func_name
    in
    let call_print_op =
      let call_op_st = OperationState.get "func.call" (Operation.loc op) in
      OperationState.add_named_attributes
        call_op_st
        [ Attribute.name "callee" (get_or_insert_print ()) ];
      OperationState.add_operands call_op_st [ Operation.result cast_op 0 ];
      Operation.create call_op_st
    in
    Block.insert_ops_after blk op [ cast_op; call_print_op ]
  | _ -> failwith "unknown op"


let run modul_op _ =
  let main = Block.first_operation @@ Region.first_block @@ Operation.region modul_op 0 in
  let main_blk = Region.first_block (Operation.region main 0) in
  let () =
    List.iter (Block.ops main_blk) ~f:(fun op ->
      let () = lower_op op main_blk in
      Operation.destroy op)
  in
  lower_func main


let lower_to_affine_pass () =
  let callbacks = { ExternalPass.empty_callbacks with run } in
  let typ_id = with_type_id_alloc (fun alloc -> TypeIDAllocator.allocate_type_id alloc) in
  ExternalPass.create
    typ_id
    ~name:"lower_to_affine"
    ~arg:""
    ~desc:"Lowering to affine"
    ~op_name:"builtin.module"
    ~dep_dialects:[]
    callbacks
