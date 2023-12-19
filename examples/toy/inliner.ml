open Base
open Mlir
open IR

let inline_calls_in_main modul_blk =
  let find_func name =
    List.find (Block.ops modul_blk) ~f:(fun op ->
      let name_attr = Operation.attribute_by_name op "sym_name" in
      let func_name = BuiltinAttributes.String.value name_attr in
      String.equal func_name name)
  in
  match find_func "main" with
  | Some main_func ->
    let main_blk = IR.Region.first_block (IR.Operation.region main_func 0) in
    let rec inline () =
      let calls =
        List.filter (Block.ops main_blk) ~f:(fun op ->
          String.equal (Operation.name op) "toy.generic_call")
      in
      if List.is_empty calls
      then ()
      else (
        List.iter calls ~f:(fun call ->
          let callee_name =
            BuiltinAttributes.FlatSymbolRef.value
              (Operation.attribute_by_name call "callee")
          in
          match find_func callee_name with
          | Some callee ->
            let callee = Operation.clone callee in
            (* for each call`s arg generate 'toy.cast' to resolve difference
               between types of callee`s args and call`s arg *)
            let casts =
              List.init (Operation.num_operands call) ~f:(fun n ->
                let call_arg = Operation.operand call n in
                let op_state =
                  OperationState.get "toy.cast" (Location.unknown Context.global_ctx)
                in
                let () = OperationState.add_operands op_state [ call_arg ] in
                let () = OperationState.add_results op_state [ Mlir_gen.typ None ] in
                Operation.create op_state)
            in
            let callee_blk = Region.first_block (Operation.region callee 0) in
            if Operation.num_operands call <> Block.num_arguments callee_blk
            then
              raise
                (Failure (Printf.sprintf "mismatch number of args for '%s'" callee_name));
            (* replace uses callee block`s argument with casts results *)
            let _ =
              List.init (Block.num_arguments callee_blk) ~f:(fun n ->
                let cast = List.nth_exn casts n in
                let cast_res = Operation.result cast 0 in
                let blk_arg = Block.argument callee_blk n in
                Value.replace_uses ~old:blk_arg ~fresh:cast_res)
            in
            (* insert casts and operations of callee block (except 'toy.return') into main *)
            let () =
              let callee_ops =
                List.filter (Block.ops callee_blk) ~f:(fun op ->
                  not (Operation.equal op (Block.terminator callee_blk)))
              in
              Block.insert_ops_after main_blk call (casts @ callee_ops)
            in
            (* replace call`s result with callee`s result *)
            let () =
              let call_res = Operation.result call 0 in
              let callee_res = Operation.operand (Block.terminator callee_blk) 0 in
              Value.replace_uses ~old:call_res ~fresh:callee_res
            in
            Operation.destroy call
          | None -> raise (Failure (Printf.sprintf "'%s' function not found" callee_name)));
        inline ())
    in
    let () = inline () in
    (* remove all functions except the main *)
    List.iter
      (List.filter (Block.ops modul_blk) ~f:(fun op ->
         let attr = Operation.attribute_by_name op "sym_name" in
         not (String.equal (BuiltinAttributes.String.value attr) "main")))
      ~f:(fun func -> Operation.destroy func)
  | None -> raise (Failure "'main' function not found")


let run modul_op pm =
  let modul_blk = Module.body @@ Module.from_op modul_op in
  try inline_calls_in_main modul_blk with
  | Failure msg ->
    Stdlib.Printf.eprintf "Inliner: %s\n" msg;
    ExternalPass.signal_failure pm


let pass () =
  let callbacks = { ExternalPass.empty_callbacks with run } in
  let typ_id = with_type_id_alloc (fun alloc -> TypeIDAllocator.allocate_type_id alloc) in
  ExternalPass.create
    typ_id
    ~name:"inline_calls_in_main"
    ~arg:""
    ~desc:"Inline calls in main function"
    ~op_name:""
    ~dep_dialects:[]
    callbacks
