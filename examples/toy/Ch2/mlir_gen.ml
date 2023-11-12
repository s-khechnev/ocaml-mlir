open Mlir
open Base
open Parser2

let global_ctx = IR.Context.create ()
let symbolTable = Hashtbl.create (module String)

let toy_handle =
  let open Ctypes in
  Foreign.foreign
    "mlirGetDialectHandle__toy__"
    (void @-> returning Stubs.Typs.DialectHandle.t)


let typ shape =
  let f64 = BuiltinTypes.Float.f64 global_ctx in
  if Array.length shape = 0
  then BuiltinTypes.Tensor.unranked f64
  else BuiltinTypes.Tensor.ranked shape f64 BuiltinAttributes.null


let mlirgen_proto name args =
  let loc = IR.Location.unknown global_ctx in
  let inputs = List.init (List.length args) ~f:(fun _ -> typ [||]) in
  let func_typ_named_attr =
    let func_typ = BuiltinTypes.Function.get global_ctx ~inputs ~results:[] in
    IR.Attribute.name
      (IR.Identifier.get global_ctx "function_type")
      (BuiltinAttributes.Type.get func_typ)
  in
  let func_name_named_attr =
    IR.Attribute.name
      (IR.Identifier.get global_ctx "sym_name")
      (BuiltinAttributes.String.get global_ctx name)
  in
  let func_op_state = IR.OperationState.get "toy.func" loc in
  let () =
    IR.OperationState.add_named_attributes
      func_op_state
      [ func_name_named_attr; func_typ_named_attr ]
  in
  let region = IR.Region.create () in
  let entry_block = IR.Block.create inputs loc in
  let () = IR.Region.append_owned_block region entry_block in
  let () = IR.OperationState.add_owned_regions func_op_state [ region ] in
  IR.Operation.create func_op_state


let rec mlirgen_expr block =
  let const_op shp values =
    let typ = typ shp in
    let attr = BuiltinAttributes.Elements.Dense.double_get typ values in
    let named_attr = IR.Attribute.name (IR.Identifier.get global_ctx "value") attr in
    let const_op_state =
      IR.OperationState.get "toy.constant" (IR.Location.unknown global_ctx)
    in
    let () = IR.OperationState.add_named_attributes const_op_state [ named_attr ] in
    let () = IR.OperationState.add_results const_op_state [ typ ] in
    IR.Operation.create const_op_state
  in
  let append_operation_and_get_result op =
    let () = IR.Block.append_owned_operation block op in
    IR.Operation.result op 0
  in
  function
  | Ast.Num n -> append_operation_and_get_result (const_op [| 0 |] [ n ])
  | Ast.Literal (shp, exs) ->
    let values =
      let rec helper acc = function
        | Ast.Num n -> n :: acc
        | Ast.Literal (_, exs) -> List.fold exs ~init:acc ~f:(fun acc e -> helper acc e)
        | _ -> assert false
      in
      helper [] (Ast.Literal ([||], exs)) |> List.rev
    in
    append_operation_and_get_result (const_op shp values)
  | Ast.VarDecl (name, shape, init_expr) ->
    let value = mlirgen_expr block init_expr in
    let op =
      if Array.length shape <> 0
      then (
        let typ = typ shape in
        let op_state =
          IR.OperationState.get "toy.reshape" (IR.Location.unknown global_ctx)
        in
        let () = IR.OperationState.add_operands op_state [ value ] in
        let () = IR.OperationState.add_results op_state [ typ ] in
        IR.Operation.create op_state)
      else IR.Value.op_result_get_owner value
    in
    let () = Hashtbl.add_exn symbolTable ~key:name ~data:(IR.Operation.result op 0) in
    append_operation_and_get_result op
  | Ast.Return expr ->
    let op_state = IR.OperationState.get "toy.return" (IR.Location.unknown global_ctx) in
    let () =
      match expr with
      | Some e ->
        let value = mlirgen_expr block e in
        IR.OperationState.add_operands op_state [ value ]
      | _ -> ()
    in
    append_operation_and_get_result (IR.Operation.create op_state)
  | Ast.BinOp (op, lhs, rhs) ->
    let lhs = mlirgen_expr block lhs in
    let rhs = mlirgen_expr block rhs in
    let op_name =
      Stdlib.Printf.sprintf
        "toy.%s"
        (match op with
         | '+' -> "add"
         | '*' -> "mul"
         | _ -> assert false)
    in
    let op_state = IR.OperationState.get op_name (IR.Location.unknown global_ctx) in
    let () = IR.OperationState.add_operands op_state [ lhs; rhs ] in
    let () = IR.OperationState.add_results op_state [ IR.Value.get_type rhs ] in
    append_operation_and_get_result (IR.Operation.create op_state)
  | Ast.Call (f_name, exprs) ->
    let operands =
      List.fold_right exprs ~init:[] ~f:(fun e acc -> mlirgen_expr block e :: acc)
    in
    (match f_name with
     | "transpose"
       when Stdlib.(
              match operands with
              | [ _ ] -> true
              | _ -> false) ->
       let op_state =
         IR.OperationState.get "toy.transpose" (IR.Location.unknown global_ctx)
       in
       let () = IR.OperationState.add_operands op_state operands in
       let () = IR.OperationState.add_results op_state [ typ [||] ] in
       append_operation_and_get_result (IR.Operation.create op_state)
     | "transpose" -> assert false
     | _ ->
       let op_state =
         IR.OperationState.get "toy.generic_call" (IR.Location.unknown global_ctx)
       in
       let callee_named_attr =
         IR.Attribute.name
           (IR.Identifier.get global_ctx "callee")
           (BuiltinAttributes.FlatSymbolRef.get global_ctx f_name)
       in
       let () = IR.OperationState.add_named_attributes op_state [ callee_named_attr ] in
       let () =
         if Stdlib.(operands <> []) then IR.OperationState.add_operands op_state operands
       in
       let () = IR.OperationState.add_results op_state [ typ [||] ] in
       append_operation_and_get_result (IR.Operation.create op_state))
  | Ast.Print expr ->
    let value = mlirgen_expr block expr in
    let op_state = IR.OperationState.get "toy.print" (IR.Location.unknown global_ctx) in
    let () = IR.OperationState.add_operands op_state [ value ] in
    append_operation_and_get_result (IR.Operation.create op_state)
  | Ast.Var var -> Hashtbl.find_exn symbolTable var


let mlirgen_func name f_args exprs =
  let () = Hashtbl.clear symbolTable in
  let func = mlirgen_proto name f_args in
  let block = IR.Region.first_block (IR.Operation.region func 0) in
  let block_args =
    List.init (IR.Block.num_arguments block) ~f:(fun n -> IR.Block.argument block n)
  in
  let () =
    List.iter2_exn f_args block_args ~f:(fun f_arg b_arg ->
      Hashtbl.add_exn symbolTable ~key:f_arg ~data:b_arg)
  in
  let () =
    List.iter exprs ~f:(fun e ->
      let _ = mlirgen_expr block e in
      ())
  in
  match IR.Operation.num_operands (IR.Block.terminator block) with
  | 0 -> func
  | _ ->
    let result_typ = typ [||] in
    let input_typs = List.map block_args ~f:(fun v -> IR.Value.get_type v) in
    let func_typ =
      BuiltinTypes.Function.get global_ctx ~inputs:input_typs ~results:[ result_typ ]
    in
    let func_attr = BuiltinAttributes.Type.get func_typ in
    IR.Operation.set_attribute_by_name func "function_type" func_attr;
    func


let mlirgen modul =
  let dhandle = toy_handle () in
  let () = IR.DialectHandle.register dhandle global_ctx in
  let _ = IR.Context.get_or_load_dialect global_ctx "toy" in
  let mlir_module = IR.Module.empty (IR.Location.unknown global_ctx) in
  let module_block = IR.Module.body mlir_module in
  let funcs =
    List.fold modul ~init:[] ~f:(fun acc f ->
      match f with
      | Ast.Function (proto, exprs) ->
        (match proto with
         | Prototype (name, args) -> mlirgen_func name args exprs :: acc))
  in
  let () =
    List.iter (List.rev funcs) ~f:(fun f ->
      IR.Block.append_owned_operation module_block f)
  in
  mlir_module
