open Parser5.Ast

let modul =
  [ Function
      ( Prototype ("multiply_transpose", [ "a"; "b" ])
      , [ Return
            (Some
               (BinOp
                  ('+', Call ("transpose", [ Var "a" ]), Call ("transpose", [ Var "b" ]))))
        ] )
  ; Function
      ( Prototype ("main", [])
      , [ (* VarDecl ("a", [| 2 |], Literal ([| 2 |], [ Num 1.0; Num 2.0 ])) *)
          VarDecl ("a", [| 2 |], Literal ([| 2 |], [ Num 1.0; Num 2.0 ]))
        ; VarDecl ("b", [||], Call ("transpose", [ Var "a" ]))
        ; VarDecl ("c", [||], BinOp ('+', Var "a", Var "b"))
        ; Print (Var "c")
        ; Return None
        ] )
  ]


open Mlir

let () =
  let dhandle = IR.DialectHandle.get "toy" in
  let () = IR.DialectHandle.register dhandle IR.Context.global_ctx in
  let _ = IR.Context.get_or_load_dialect IR.Context.global_ctx "toy" in
  let mlir_modul = Mlir_gen.mlirgen modul in
  let () = Inliner.inline_calls_in_main mlir_modul in
  let () =
    let infer_pass = Shape_inference.infer_shapes_pass () in
    let cse_pass = Transforms.CSE.create () in
    let canon_pass = Transforms.Canonicalizer.create () in
    let pm = PassManager.create IR.Context.global_ctx in
    let op_pm = PassManager.nested_under pm "toy.func" in
    let () = OpPassManager.add_owned_pass op_pm infer_pass in
    let () = OpPassManager.add_owned_pass op_pm canon_pass in
    let () = PassManager.add_owned_pass pm cse_pass in
    let _ = PassManager.run pm mlir_modul in
    ()
  in
  IR.Operation.dump (IR.Module.operation mlir_modul);
  let reg = IR.DialectRegistry.create () in
  let () = RegisterEverything.dialects reg in
  let () = IR.Context.append_dialect_registry IR.Context.global_ctx reg in
  let _ = IR.Context.get_or_load_dialect IR.Context.global_ctx "affine" in
  let _ = IR.Context.get_or_load_dialect IR.Context.global_ctx "memref" in
  let _ = IR.Context.get_or_load_dialect IR.Context.global_ctx "arith" in
  let _ = IR.Context.get_or_load_dialect IR.Context.global_ctx "func" in
  Lower_to_affine.lower mlir_modul;
  IR.Operation.dump @@ IR.Module.operation mlir_modul
