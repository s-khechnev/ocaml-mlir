open Mlir
open IR
open Base

(* The ShapeInferencePass is a pass that performs intra-procedural
   shape inference.

   Algorithm:

   1) Build a worklist containing all the operations that return a
   dynamically shaped tensor: these are the operations that need shape
   inference.
   2) Iterate on the worklist:
   a) find an operation to process: the next ready operation in the
   worklist has all of its arguments non-generic,
   b) if no operation is found, break out of the loop,
   c) remove the operation from the worklist,
   d) infer the shape of its output from the argument types.
   3) If the worklist is empty, the algorithm succeeded. *)

let run main_func pm =
  let main_blk = Region.first_block (Operation.region main_func 0) in
  (* Populate the worklist with the operations that need shape inference:
     these are operations that return a dynamic shape. *)
  let worklist =
    let returns_dynamic_shape op =
      if Operation.num_results op = 0
      then false
      else (
        let result = Operation.result op 0 in
        BuiltinTypes.Tensor.is_unranked_tensor (Value.get_type result))
    in
    List.filter (Block.ops main_blk) ~f:returns_dynamic_shape
  in
  let all_operands_inferred op =
    let opers = List.init (Operation.num_operands op) ~f:(Operation.operand op) in
    List.for_all opers ~f:(fun value ->
      BuiltinTypes.Tensor.is_ranked_tensor (Value.get_type value))
  in
  (* Iterate on the operations in the worklist until all operations have been
     inferred or no change happened (fix point). *)
  let rec loop lst =
    if List.is_empty lst
    then ()
    else (
      (* Find the next operation ready for inference, that is an operation
         with all operands already resolved (non-generic). *)
      match List.find lst ~f:all_operands_inferred with
      | Some op ->
        let result_shape =
          match Operation.name op with
          | "toy.add" | "toy.mul" | "toy.cast" -> Value.get_type @@ Operation.operand op 0
          | "toy.transpose" ->
            let typ = Value.get_type @@ Operation.operand op 0 in
            let rank = BuiltinTypes.Shaped.rank typ in
            let transposed_shape =
              Array.rev @@ Array.init rank ~f:(BuiltinTypes.Shaped.dim_size typ)
            in
            Mlir_gen.typ (Some transposed_shape)
          | _ -> raise (Failure "unknown operation")
        in
        (* Change the type of the operation result. In fact, create a similar operation,
           but with a different type of results and insert it instead of the old operation,
           deleting the old one. *)
        let () =
          let opers = List.init (Operation.num_operands op) ~f:(Operation.operand op) in
          let new_op =
            let op_st = OperationState.get (Operation.name op) (Operation.loc op) in
            let () = OperationState.add_operands op_st opers in
            let () = OperationState.add_results op_st [ result_shape ] in
            Operation.create op_st
          in
          Value.replace_uses
            ~old:(Operation.result op 0)
            ~fresh:(Operation.result new_op 0);
          Block.insert_owned_operation_after main_blk op new_op;
          Operation.destroy op
        in
        (* Remove inferred op from lst *)
        let lst = List.filter lst ~f:(fun opf -> not (Operation.equal opf op)) in
        loop lst
      | None ->
        if not (List.is_empty lst) then raise (Failure "operations couldn't be inferred"))
  in
  try
    loop worklist;
    (* remove casts *)
    List.iter (Block.ops main_blk) ~f:(fun op ->
      if String.equal (Operation.name op) "toy.cast"
      then (
        Value.replace_uses ~old:(Operation.result op 0) ~fresh:(Operation.operand op 0);
        Operation.destroy op))
  with
  | Failure msg ->
    Stdlib.Printf.eprintf "Shape inference: %s\n" msg;
    ExternalPass.signal_failure pm


let pass () =
  let callbacks = { ExternalPass.empty_callbacks with run } in
  let typ_id = with_type_id_alloc (fun alloc -> TypeIDAllocator.allocate_type_id alloc) in
  ExternalPass.create
    typ_id
    ~name:"shape_inference"
    ~arg:""
    ~desc:"Inference the tensor's shape"
    ~op_name:"toy.func"
    ~dep_dialects:[]
    callbacks
