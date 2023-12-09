open Base
open Mlir

let memref_of_tensor tensor =
  let () = assert (BuiltinTypes.Tensor.is_ranked_tensor tensor) in
  let rank = BuiltinTypes.Shaped.rank tensor in
  let shape = Array.init rank ~f:(fun dim -> BuiltinTypes.Shaped.dim_size tensor dim) in
  BuiltinTypes.MemRef.get
    (BuiltinTypes.Float.f64 IR.Context.global_ctx)
    shape
    BuiltinAttributes.null
    BuiltinAttributes.null


let alloc_dealloc typ loc blk =
  let alloc =
    let alloc_st = IR.OperationState.get "memref.alloc" loc in
    let () =
      IR.OperationState.add_named_attributes
        alloc_st
        [ IR.Attribute.name
            "operand_segment_sizes"
            (BuiltinAttributes.Dense.Array.i32 IR.Context.global_ctx [ 0; 0 ])
        ]
    in
    let () = IR.OperationState.add_results alloc_st [ typ ] in
    IR.Operation.create alloc_st
  in
  let fst_op = IR.Block.first_operation blk in
  let () = IR.Block.insert_owned_operation_before blk fst_op alloc in
  let alloc_val = IR.Operation.result alloc 0 in
  let dealloc =
    let dealloc_st = IR.OperationState.get "memref.dealloc" loc in
    let () = IR.OperationState.add_operands dealloc_st [ alloc_val ] in
    IR.Operation.create dealloc_st
  in
  let term_op = IR.Block.terminator blk in
  let () = IR.Block.insert_owned_operation_before blk term_op dealloc in
  alloc_val


let map mref_typ =
  let rank = BuiltinTypes.Shaped.rank mref_typ in
  if rank = 0
  then AffineMap.empty IR.Context.global_ctx
  else AffineMap.multi_dim_identity IR.Context.global_ctx rank


let affine_store_op loc value_to_store memref indices =
  let mref_typ = IR.Value.get_type memref in
  let map = map mref_typ in
  let affine_store_op_st = IR.OperationState.get "affine.store" loc in
  IR.OperationState.add_operands affine_store_op_st ([ value_to_store; memref ] @ indices);
  IR.OperationState.add_named_attributes
    affine_store_op_st
    [ IR.Attribute.name "map" (BuiltinAttributes.AffineMap.get map) ];
  IR.Operation.create affine_store_op_st


let affine_load_op loc ivs alloc =
  let affine_load_op_st = IR.OperationState.get "affine.load" loc in
  let mref_typ = IR.Value.get_type alloc in
  let map = map mref_typ in
  IR.OperationState.add_named_attributes
    affine_load_op_st
    [ IR.Attribute.name "map" (BuiltinAttributes.AffineMap.get map) ];
  IR.OperationState.add_operands affine_load_op_st [ alloc ];
  IR.OperationState.add_operands affine_load_op_st ivs;
  IR.OperationState.add_results
    affine_load_op_st
    [ BuiltinTypes.Float.f64 IR.Context.global_ctx ];
  IR.Operation.create affine_load_op_st


let build_loop_from_consts loc lb ub body_builder =
  let lb_map = AffineMap.constant IR.Context.global_ctx lb in
  let ub_map = AffineMap.constant IR.Context.global_ctx ub in
  let for_op_st = IR.OperationState.get "affine.for" loc in
  let step_attr =
    let step =
      BuiltinAttributes.Integer.get (BuiltinTypes.Index.get IR.Context.global_ctx) 1
    in
    IR.Attribute.name "step" step
  in
  let bound_attr name v = IR.Attribute.name name (BuiltinAttributes.AffineMap.get v) in
  let () =
    IR.OperationState.add_named_attributes
      for_op_st
      [ step_attr; bound_attr "lower_bound" lb_map; bound_attr "upper_bound" ub_map ]
  in
  let region = IR.Region.create () in
  let entry_block =
    IR.Block.create [ BuiltinTypes.Index.get IR.Context.global_ctx ] loc
  in
  let () = IR.Region.append_owned_block region entry_block in
  let () = IR.OperationState.add_owned_regions for_op_st [ region ] in
  let induction_var = IR.Block.argument entry_block 0 in
  let op = IR.Operation.create for_op_st in
  let () = body_builder loc entry_block induction_var in
  op


let build_loop_nest loc lower_bounds upper_bounds body_builder =
  let () = assert (Array.(length lower_bounds = length upper_bounds)) in
  let count = Array.length lower_bounds in
  if count = 0
  then (
    let dummy_blk = IR.Block.create [] loc in
    body_builder dummy_blk loc [];
    IR.Block.ops dummy_blk)
  else (
    let ivs = Queue.create () in
    let rec loop i ?parent_blk ops =
      if i = count
      then (
        List.iter ops ~f:(fun op ->
          let op_blk = IR.Region.first_block @@ IR.Operation.region op 0 in
          let yield_op =
            IR.Operation.create @@ IR.OperationState.get "affine.yield" loc
          in
          IR.Block.append_owned_operation op_blk yield_op);
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
          | Some blk -> IR.Block.append_owned_operation blk for_op
          | None -> ()
        in
        let for_blk = IR.Region.first_block @@ IR.Operation.region for_op 0 in
        loop (i + 1) ~parent_blk:for_blk (for_op :: ops))
    in
    loop 0 [])


let lower_op_to_loops op blk body_builder =
  let tensor_typ = IR.Value.get_type @@ IR.Operation.result op 0 in
  let loc = IR.Operation.loc op in
  let mref_typ = memref_of_tensor tensor_typ in
  let alloc = alloc_dealloc mref_typ loc blk in
  let rank = BuiltinTypes.Shaped.rank tensor_typ in
  let lower_bounds = Array.init rank ~f:(fun _ -> 0) in
  let shape =
    Array.init rank ~f:(fun dim -> BuiltinTypes.Shaped.dim_size tensor_typ dim)
  in
  let body_builder blk loc ivs =
    let value_to_store = body_builder blk loc ivs in
    let op = IR.Value.op_result_get_owner value_to_store in
    let blk = IR.Operation.block op in
    let affine_store = affine_store_op loc value_to_store alloc ivs in
    IR.Block.insert_owned_operation_after blk op affine_store
  in
  IR.Value.replace_uses ~old:(IR.Operation.result op 0) ~fresh:alloc;
  let ops = build_loop_nest loc lower_bounds shape body_builder in
  IR.Block.insert_ops_after blk op ops


let lower_bin_op op blk =
  let body_builder blk loc ivs =
    let lhs, rhs = IR.Operation.(operand op 0, operand op 1) in
    let affine_load_op = affine_load_op loc ivs in
    let loaded_lhs, loaded_rhs = affine_load_op lhs, affine_load_op rhs in
    let lowered_bin_op =
      let bin_op name =
        let add_op_st = IR.OperationState.get name loc in
        IR.OperationState.add_operands
          add_op_st
          IR.Operation.[ result loaded_lhs 0; result loaded_rhs 0 ];
        IR.OperationState.add_results
          add_op_st
          [ BuiltinTypes.Float.f64 IR.Context.global_ctx ];
        IR.Operation.create add_op_st
      in
      match IR.Operation.name op with
      | "toy.add" -> bin_op "arith.addf"
      | "toy.mul" -> bin_op "arith.msulf"
      | _ -> assert false
    in
    IR.Block.append_owned_operation blk loaded_lhs;
    IR.Block.append_owned_operation blk loaded_rhs;
    IR.Block.append_owned_operation blk lowered_bin_op;
    IR.Operation.result lowered_bin_op 0
  in
  lower_op_to_loops op blk body_builder


let lower_const_op op blk =
  let loc = IR.Operation.loc op in
  let value_attr = IR.Operation.attribute_by_name op "value" in
  let tensor_typ = IR.Value.get_type @@ IR.Operation.result op 0 in
  let mref_typ = memref_of_tensor tensor_typ in
  let alloc = alloc_dealloc mref_typ loc blk in
  let value_shape =
    List.init (BuiltinTypes.Shaped.rank mref_typ) ~f:(fun n ->
      BuiltinTypes.Shaped.dim_size mref_typ n)
  in
  let build_const_idx idx =
    let const_id_st = IR.OperationState.get "arith.constant" loc in
    let idx =
      BuiltinAttributes.Integer.get (BuiltinTypes.Index.get IR.Context.global_ctx) idx
    in
    let () =
      IR.OperationState.add_named_attributes const_id_st [ IR.Attribute.name "value" idx ]
    in
    let () =
      IR.OperationState.add_results
        const_id_st
        [ BuiltinTypes.Index.get IR.Context.global_ctx ]
    in
    IR.Operation.create const_id_st
  in
  let const_indices =
    if List.is_empty value_shape
    then [| build_const_idx 0 |]
    else (
      let max_dim = Option.value_exn @@ List.max_elt value_shape ~compare:Int.compare in
      Array.init max_dim ~f:(fun n -> build_const_idx n))
  in
  let coords =
    let rec gen_coords shape acc =
      match shape with
      | [] -> [ List.rev acc ]
      | dim :: rest ->
        List.concat_map (List.init dim ~f:Fn.id) ~f:(fun c -> gen_coords rest (c :: acc))
    in
    gen_coords value_shape []
  in
  let ops =
    List.fold coords ~init:[] ~f:(fun acc coord ->
      let const_op =
        let const_op_st = IR.OperationState.get "arith.constant" loc in
        let value =
          let value =
            BuiltinAttributes.Float.value
              (BuiltinAttributes.Elements.get value_attr coord)
          in
          BuiltinAttributes.Float.get
            IR.Context.global_ctx
            (BuiltinTypes.Float.f64 IR.Context.global_ctx)
            value
        in
        IR.OperationState.add_named_attributes
          const_op_st
          [ IR.Attribute.name "value" value ];
        IR.OperationState.add_results
          const_op_st
          [ BuiltinTypes.Float.f64 IR.Context.global_ctx ];
        IR.Operation.create const_op_st
      in
      let indices =
        List.map coord ~f:(fun i ->
          let const_op = const_indices.(i) in
          IR.Operation.result const_op 0)
      in
      let affine_store_op =
        affine_store_op loc (IR.Operation.result const_op 0) alloc indices
      in
      [ const_op; affine_store_op ] @ acc)
  in
  IR.Value.replace_uses ~old:(IR.Operation.result op 0) ~fresh:alloc;
  IR.Block.insert_ops_after blk op (Array.to_list const_indices @ ops)


let lower_return op blk =
  if IR.Operation.num_operands op <> 0 then failwith "Return must not have an operand";
  let func_ret_op =
    IR.Operation.create @@ IR.OperationState.get "func.return" (IR.Operation.loc op)
  in
  IR.Block.insert_owned_operation_before blk op func_ret_op


let lower_func op =
  if IR.Operation.(num_operands op <> 0 || num_results op <> 0)
  then failwith "expected 'main' to have 0 inputs and 0 results";
  let func_op =
    let op = IR.Operation.clone op in
    let func_op_st = IR.OperationState.get "func.func" (IR.Operation.loc op) in
    let attrs =
      List.init (IR.Operation.num_attributes op) ~f:(fun n -> IR.Operation.attribute op n)
    in
    IR.OperationState.add_named_attributes func_op_st attrs;
    let reg = IR.Region.create () in
    let main_blk = IR.Region.first_block (IR.Operation.region op 0) in
    IR.Region.append_owned_block reg main_blk;
    IR.OperationState.add_owned_regions func_op_st [ reg ];
    IR.Operation.create func_op_st
  in
  let modul_blk = IR.Operation.block op in
  IR.Block.append_owned_operation modul_blk func_op;
  IR.Operation.destroy op


let change_print op blk =
  let print_op =
    let print_op_st = IR.OperationState.get "toy.print" (IR.Operation.loc op) in
    let () = IR.OperationState.add_operands print_op_st [ IR.Operation.operand op 0 ] in
    IR.Operation.create print_op_st
  in
  IR.Block.insert_owned_operation_after blk op print_op


let lower_transpose op blk =
  let body_builder blk loc ivs =
    let input = IR.Operation.operand op 0 in
    let ivs = List.rev ivs in
    let load = affine_load_op loc ivs input in
    IR.Block.append_owned_operation blk load;
    IR.Operation.result load 0
  in
  lower_op_to_loops op blk body_builder


let lower_op op blk =
  match IR.Operation.name op with
  | "toy.constant" -> lower_const_op op blk
  | "toy.add" | "toy.mul" -> lower_bin_op op blk
  | "toy.return" -> lower_return op blk
  | "toy.print" -> change_print op blk
  | "toy.transpose" -> lower_transpose op blk
  | _ -> failwith "unknown op"


let run modul_op _ =
  let main =
    IR.Block.first_operation @@ IR.Region.first_block @@ IR.Operation.region modul_op 0
  in
  let main_blk = IR.Region.first_block (IR.Operation.region main 0) in
  let () =
    List.iter (IR.Block.ops main_blk) ~f:(fun op ->
      lower_op op main_blk;
      IR.Operation.destroy op)
  in
  lower_func main


let lower_to_affine_pass () =
  let callbacks = { ExternalPass.empty_callbacks with run } in
  let alloc = TypeIDAllocator.create () in
  let typ_id = TypeIDAllocator.allocate_type_id alloc in
  ExternalPass.create
    typ_id
    ~name:"lower_to_affine"
    ~arg:""
    ~desc:"Lowering to affine"
    ~op_name:""
    ~dep_dialects:[]
    callbacks
