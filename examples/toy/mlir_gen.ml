open Mlir
open IR
open Base
open Parse

let symbols = Hashtbl.create (module String)

(* fix me *)
let loc = Location.unknown Context.global_ctx

let typ =
  let f64 = BuiltinTypes.Float.f64 Context.global_ctx in
  function
  | Some shape -> BuiltinTypes.Tensor.ranked shape f64 BuiltinAttributes.null
  | None -> BuiltinTypes.Tensor.unranked f64


let rec mlirgen_expr block =
  let mlirgen_expr e = mlirgen_expr block e in
  let append_op_get_result op =
    let () = Block.append_owned_operation block op in
    Operation.result op 0
  in
  let const_op shp values =
    let const_op_st = OperationState.get "toy.constant" loc in
    let typ = typ (Some shp) in
    let value_attr =
      let value_attr = BuiltinAttributes.Dense.Elements.double_get typ values in
      Attribute.name "value" value_attr
    in
    let () = OperationState.add_results const_op_st [ typ ] in
    let () = OperationState.add_named_attributes const_op_st [ value_attr ] in
    Operation.create const_op_st
  in
  function
  | Ast.Num n -> append_op_get_result (const_op [||] [ n ])
  | Ast.Literal (shape, exs) ->
    let values =
      let rec helper acc = function
        | Ast.Num n -> n :: acc
        | Ast.Literal (_, exs) -> List.fold exs ~init:acc ~f:(fun acc e -> helper acc e)
        | _ -> assert false
      in
      helper [] (Ast.Literal ([||], exs)) |> List.rev
    in
    append_op_get_result (const_op shape values)
  | Ast.VarDecl (name, shape, init_expr) ->
    let init_value = mlirgen_expr init_expr in
    let value =
      if Array.length shape <> 0
      then (
        let reshape_op =
          let typ = typ (Some shape) in
          let op_st = OperationState.get "toy.reshape" loc in
          let () = OperationState.add_operands op_st [ init_value ] in
          let () = OperationState.add_results op_st [ typ ] in
          Operation.create op_st
        in
        append_op_get_result reshape_op)
      else init_value
    in
    (match Hashtbl.add symbols ~key:name ~data:value with
     | `Duplicate -> failwith @@ Printf.sprintf "'%s' already defined." name
     | `Ok -> value)
  | Ast.Return expr ->
    let ret_op =
      let op_st = OperationState.get "toy.return" loc in
      (match expr with
       | Some e ->
         let value = mlirgen_expr e in
         OperationState.add_operands op_st [ value ]
       | _ -> ());
      Operation.create op_st
    in
    append_op_get_result ret_op
  | Ast.BinOp (op, lhs, rhs) ->
    let lhs = mlirgen_expr lhs in
    let rhs = mlirgen_expr rhs in
    let op_name =
      Printf.sprintf
        "toy.%s"
        (match op with
         | `Add -> "add"
         | `Mul -> "mul")
    in
    let bin_op =
      let op_st = OperationState.get op_name loc in
      let () = OperationState.add_operands op_st [ lhs; rhs ] in
      let () = OperationState.add_results op_st [ Value.get_type lhs ] in
      Operation.create op_st
    in
    append_op_get_result bin_op
  | Ast.Call (f_name, exprs) ->
    let operands =
      List.fold_right exprs ~init:[] ~f:(fun e acc -> mlirgen_expr e :: acc)
    in
    (match f_name with
     | "transpose" ->
       (match operands with
        | [ _ ] ->
          let transpose_op =
            let op_st = OperationState.get "toy.transpose" loc in
            let () = OperationState.add_operands op_st operands in
            let () = OperationState.add_results op_st [ typ None ] in
            Operation.create op_st
          in
          append_op_get_result transpose_op
        | _ -> failwith "transpose must have 1 argument")
     | _ ->
       let call_op =
         let op_st = OperationState.get "toy.generic_call" loc in
         let callee_attr =
           Attribute.name
             "callee"
             (BuiltinAttributes.FlatSymbolRef.get Context.global_ctx f_name)
         in
         let () = OperationState.add_named_attributes op_st [ callee_attr ] in
         let () = OperationState.add_operands op_st operands in
         let () = OperationState.add_results op_st [ typ None ] in
         Operation.create op_st
       in
       append_op_get_result call_op)
  | Ast.Print expr ->
    let value = mlirgen_expr expr in
    let print_op =
      let op_state = OperationState.get "toy.print" loc in
      let () = OperationState.add_operands op_state [ value ] in
      Operation.create op_state
    in
    append_op_get_result print_op
  | Ast.Var var ->
    (match Hashtbl.find symbols var with
     | Some v -> v
     | None -> failwith @@ Printf.sprintf "unknown variable: %s" var)


let mlirgen_func = function
  | Ast.Function (proto, exprs) ->
    (match proto with
     | Ast.Prototype (f_name, f_args) ->
       let () = Hashtbl.clear symbols in
       let num_args = List.length f_args in
       let arg_typs = List.init num_args ~f:(fun _ -> typ None) in
       let blk = Block.create arg_typs (List.init num_args ~f:(fun _ -> loc)) in
       (* add block's args in symbol table *)
       let () =
         let block_args = List.init num_args ~f:(Block.argument blk) in
         List.iter2_exn f_args block_args ~f:(fun name value ->
           Hashtbl.add_exn symbols ~key:name ~data:value)
       in
       let () =
         List.iter exprs ~f:(fun e ->
           let _ = mlirgen_expr blk e in
           ());
         if Operation.is_null @@ Block.terminator blk
         then (
           let _ = mlirgen_expr blk (Return None) in
           ())
       in
       let func_op_state = OperationState.get "toy.func" loc in
       let attrs =
         let func_typ_attr =
           let results =
             match Operation.num_operands @@ Block.terminator blk with
             | 0 -> []
             | _ -> [ typ None ]
           in
           let func_typ =
             BuiltinTypes.Function.get Context.global_ctx ~inputs:arg_typs ~results
           in
           Attribute.name "function_type" (BuiltinAttributes.Type.get func_typ)
         in
         let func_name_attr =
           Attribute.name
             "sym_name"
             (BuiltinAttributes.String.get Context.global_ctx f_name)
         in
         [ func_name_attr; func_typ_attr ]
       in
       let () = OperationState.add_named_attributes func_op_state attrs in
       let region = Region.create () in
       let () = Region.append_owned_block region blk in
       let () = OperationState.add_owned_regions func_op_state [ region ] in
       Operation.create func_op_state)


let mlirgen modul =
  let mlir_module = Module.empty loc in
  let module_block = Module.body mlir_module in
  let () =
    List.iter modul ~f:(fun f ->
      let f = mlirgen_func f in
      Block.append_owned_operation module_block f)
  in
  mlir_module
