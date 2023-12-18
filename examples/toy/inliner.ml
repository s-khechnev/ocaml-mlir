open Base
open Mlir

let rec find_func_by_name op name =
  if IR.Operation.is_null op
  then raise Stdlib.Not_found
  else (
    let name_attr = IR.Operation.attribute_by_name op "sym_name" in
    let func_name = BuiltinAttributes.String.value name_attr in
    if String.equal func_name name
    then op
    else find_func_by_name (IR.Operation.next_in_block op) name)


let inline_calls_in_main modul =
  let modul_blk = IR.Module.body modul in
  let fst_func = IR.Block.first_operation modul_blk in
  let find_func = find_func_by_name fst_func in
  let main_func = find_func "main" in
  let main_block = IR.Region.first_block (IR.Operation.region main_func 0) in
  let rec inline () =
    let calls =
      List.filter (IR.Block.ops main_block) ~f:(fun op ->
        String.equal (IR.Operation.name op) "toy.generic_call")
    in
    match calls with
    | [] -> ()
    | _ ->
      let () =
        List.iter calls ~f:(fun call ->
          let call_args =
            List.init (IR.Operation.num_operands call) ~f:(fun n ->
              IR.Operation.operand call n)
          in
          (* for each call`s arg we generate toy.cast for resolve difference
             between types of callee`s args and call`s arg *)
          let casts =
            List.rev
            @@ List.fold call_args ~init:[] ~f:(fun acc arg ->
              let op_state =
                IR.OperationState.get
                  "toy.cast"
                  (IR.Location.unknown IR.Context.global_ctx)
              in
              let () = IR.OperationState.add_operands op_state [ arg ] in
              let () = IR.OperationState.add_results op_state [ Mlir_gen.typ None ] in
              IR.Operation.create op_state :: acc)
          in
          let callee =
            let callee_name =
              BuiltinAttributes.FlatSymbolRef.value
                (IR.Operation.attribute_by_name call "callee")
            in
            find_func callee_name
          in
          (* clone callee *)
          let callee = IR.Operation.clone callee in
          let callee_blk = IR.Region.first_block (IR.Operation.region callee 0) in
          (* replace uses callee block`s argument with casts results *)
          let () =
            let args =
              List.init (IR.Block.num_arguments callee_blk) ~f:(fun n ->
                let cast_res = IR.Operation.result (List.nth_exn casts n) 0 in
                let blk_arg = IR.Block.argument callee_blk n in
                blk_arg, cast_res)
            in
            List.iter args ~f:(fun (arg, replacmnt) ->
              IR.Value.replace_uses ~old:arg ~fresh:replacmnt)
          in
          (* insert casts and operations of callee block into main *)
          let callee_ops =
            List.filter (IR.Block.ops callee_blk) ~f:(fun op ->
              not (IR.Operation.equal op (IR.Block.terminator callee_blk)))
          in
          let () = IR.Block.insert_ops_after main_block call (casts @ callee_ops) in
          (* replace call`s result with callee`s result *)
          let call_res = IR.Operation.result call 0 in
          let callee_res = IR.Operation.operand (IR.Block.terminator callee_blk) 0 in
          let () = IR.Value.replace_uses ~old:call_res ~fresh:callee_res in
          IR.Operation.destroy call)
      in
      inline ()
  in
  let () = inline () in
  (* remove all functions except the main *)
  List.iter
    (List.filter (IR.Block.ops modul_blk) ~f:(fun op ->
       let attr = IR.Operation.attribute_by_name op "sym_name" in
       not (String.equal (BuiltinAttributes.String.value attr) "main")))
    ~f:(fun func -> IR.Operation.destroy func)
