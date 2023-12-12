(*
   ```mlir
  %0 = rgn.val { return 3 }
  %1 = rgn.val { return 5 }
  %2 = select true, %0, %1
  rgn.run %2
```
*)
open Mlir

let () =
  let ctx = IR.Context.global_ctx in
  let _ =
    let rgn = "rgn" in
    let rgn_handle = IR.DialectHandle.get rgn in
    let () = IR.DialectHandle.register rgn_handle ctx in
    IR.Context.get_or_load_dialect ctx rgn
  in
  let _ =
    let registry = IR.DialectRegistry.create () in
    let () = RegisterEverything.dialects registry in
    let () = IR.Context.append_dialect_registry ctx registry in
    IR.Context.get_or_load_dialect ctx "arith"
  in
  let dummy_loc = IR.Location.unknown ctx in
  let modul = IR.Module.empty dummy_loc in
  let append ops =
    let modul_blk = IR.Module.body modul in
    List.iter (IR.Block.append_owned_operation modul_blk) ops
  in
  let i32 = BuiltinTypes.Integer.get ctx 32 in
  let const_op bit_width v =
    let const_op_st = IR.OperationState.get "arith.constant" dummy_loc in
    let value = BuiltinAttributes.Integer.get bit_width v in
    IR.OperationState.add_named_attributes const_op_st [ IR.Attribute.name "value" value ];
    IR.OperationState.add_results const_op_st [ bit_width ];
    IR.Operation.create const_op_st
  in
  let val_op const =
    let val_op_st = IR.OperationState.get "rgn.val" dummy_loc in
    let rgn = IR.Region.create () in
    let blk = IR.Block.create [] [ dummy_loc ] in
    IR.Region.append_owned_block rgn blk;
    let const_op = const_op i32 const in
    IR.Block.append_owned_operation blk const_op;
    let ret_op =
      let ret_op_st = IR.OperationState.get "rgn.return" dummy_loc in
      IR.OperationState.add_operands ret_op_st [ IR.Operation.result const_op 0 ];
      IR.Operation.create ret_op_st
    in
    IR.Block.append_owned_operation blk ret_op;
    IR.OperationState.add_owned_regions val_op_st [ rgn ];
    IR.OperationState.add_results val_op_st [ i32 ];
    IR.Operation.create val_op_st
  in
  let x, y = val_op 3, val_op 5 in
  let cond = const_op (BuiltinTypes.Integer.get ctx 1) 1 in
  let select_op =
    let select_op_st = IR.OperationState.get "arith.select" dummy_loc in
    let cond_val, true_val, false_val =
      IR.Operation.(result cond 0, result x 0, result y 0)
    in
    IR.OperationState.add_operands select_op_st [ cond_val; true_val; false_val ];
    IR.OperationState.add_results select_op_st [ i32 ];
    IR.Operation.create select_op_st
  in
  let run_op =
    let run_op_st = IR.OperationState.get "rgn.run" dummy_loc in
    IR.OperationState.add_operands run_op_st [ IR.Operation.result select_op 0 ];
    IR.Operation.create run_op_st
  in
  append [ x; y; cond; select_op; run_op ];
  IR.Operation.dump @@ IR.Module.operation modul
