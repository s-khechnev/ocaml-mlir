open Mlir
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

let run main_func _ =
  let main_blk = IR.Region.first_block (IR.Operation.region main_func 0) in
  (* Populate the worklist with the operations that need shape inference:
     these are operations that return a dynamic shape. *)
  let worklist =
    let returns_dynamic_shape op =
      let result = IR.Operation.result op 0 in
      BuiltinTypes.Tensor.is_unranked_tensor (IR.Value.get_type result)
    in
    List.filter (IR.Block.ops main_blk) ~f:(fun op ->
      IR.Operation.num_results op <> 0 && returns_dynamic_shape op)
  in
  let all_operands_inferred op =
    let opers =
      List.init (IR.Operation.num_operands op) ~f:(fun n -> IR.Operation.operand op n)
    in
    List.for_all opers ~f:(fun value ->
      BuiltinTypes.Tensor.is_ranked_tensor (IR.Value.get_type value))
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
          match IR.Operation.name op with
          | "toy.add" ->
            let lhs = IR.Operation.operand op 0 in
            IR.Value.get_type lhs
          | "toy.mul" ->
            let lhs = IR.Operation.operand op 0 in
            IR.Value.get_type lhs
          | "toy.cast" ->
            let oper = IR.Operation.operand op 0 in
            IR.Value.get_type oper
          | "toy.transpose" ->
            let oper = IR.Operation.operand op 0 in
            let typ = IR.Value.get_type oper in
            let rank = BuiltinTypes.Shaped.rank typ in
            let transposed_shape =
              Array.rev
              @@ Array.init rank ~f:(fun dim -> BuiltinTypes.Shaped.dim_size typ dim)
            in
            Mlir_gen.typ transposed_shape
          | _ -> failwith "unkown operation"
        in
        (* Change the type of the operation result. In fact, create a similar operation,
           but with a different type of results and insert it instead of the old operation,
           deleting the old one. *)
        let () =
          let opers =
            List.init (IR.Operation.num_operands op) ~f:(IR.Operation.operand op)
          in
          let op_state =
            IR.OperationState.get (IR.Operation.name op) (IR.Operation.loc op)
          in
          let () = IR.OperationState.add_operands op_state opers in
          let () = IR.OperationState.add_results op_state [ result_shape ] in
          let new_op = IR.Operation.create op_state in
          IR.Value.replace_uses
            ~old:(IR.Operation.result op 0)
            ~fresh:(IR.Operation.result new_op 0);
          IR.Block.insert_owned_operation_after main_blk op new_op;
          IR.Operation.destroy op
        in
        (* Remove infered op from lst *)
        let lst =
          List.filter lst ~f:(fun curr_op -> not (IR.Operation.equal curr_op op))
        in
        loop lst
      | None -> ())
  in
  loop worklist


let infer_shapes modul =
  let callbacks = { ExternalPass.empty_callbacks with run } in
  let alloc = TypeIDAllocator.create () in
  let typ_id = TypeIDAllocator.allocate_type_id alloc in
  let pass =
    ExternalPass.create
      typ_id
      ~name:"shape_inference"
      ~arg:""
      ~desc:"Inference the tensor's shape"
      ~op_name:""
      ~dep_dialects:[]
      callbacks
  in
  let pm = PassManager.create IR.Context.global_ctx in
  let op_pm = PassManager.nested_under pm "toy.func" in
  let () = OpPassManager.add_owned_pass op_pm pass in
  let _ = PassManager.run pm modul in
  ()
