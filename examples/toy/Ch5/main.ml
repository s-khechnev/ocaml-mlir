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
      , [ (* VarDecl
             ( "a"
             , [| 2; 3 |]
             , Literal
             ( [| 2; 3 |]
             , [ Literal ([| 3 |], [ Num 1.0; Num 2.0; Num 3.0 ])
                  ; Literal ([| 3 |], [ Num 4.0; Num 5.0; Num 6.0 ])
                  ] ) )
             ; VarDecl
             ( "b"
             , [| 2; 3 |]
             , Literal
             ( [| 6 |]
             , [ Literal
                      ([| 6 |], [ Num 1.0; Num 2.0; Num 3.0; Num 4.0; Num 5.0; Num 6.0 ])
                  ] ) )
             ; VarDecl ("c", [||], Call ("multiply_transpose", [ Var "a"; Var "b" ])) *)
          VarDecl ("a", [||], Num 1.0)
        ; VarDecl ("b", [||], Num 1.0)
        ; VarDecl ("c", [||], BinOp ('+', Var "a", Var "b"))
        ; Print (Var "c")
        ; Return None
        ] )
  ]


open Mlir

let () =
  let dhandle = IR.DialectHandle.get "toy" in
  let registry = IR.DialectRegistry.create () in
  let () = IR.DialectHandle.register dhandle IR.Context.global_ctx in
  let () = RegisterEverything.dialects registry in
  let () = IR.Context.append_dialect_registry IR.Context.global_ctx registry in
  let _ = IR.Context.get_or_load_dialect IR.Context.global_ctx "toy" in
  let _ = IR.Context.get_or_load_dialect IR.Context.global_ctx "affine" in
  let _ = IR.Context.get_or_load_dialect IR.Context.global_ctx "memref" in
  let _ = IR.Context.get_or_load_dialect IR.Context.global_ctx "arith" in
  let _ = IR.Context.get_or_load_dialect IR.Context.global_ctx "func" in
  let mlir_modul = Mlir_gen.mlirgen modul in
  let () = Inliner.inline_calls_in_main mlir_modul in
  let () =
    let infer_pass = Shape_inference.infer_shapes_pass () in
    let canon_pass = Transforms.Canonicalizer.create () in
    let cse_pass = Transforms.CSE.create () in
    let lower_pass = Lower_to_affine.lower_to_affine_pass () in
    let pm = PassManager.create IR.Context.global_ctx in
    let op_pm = PassManager.nested_under pm "toy.func" in
    let () = OpPassManager.add_owned_pass op_pm infer_pass in
    let () = OpPassManager.add_owned_pass op_pm canon_pass in
    let () = OpPassManager.add_owned_pass op_pm cse_pass in
    let () = OpPassManager.add_owned_pass op_pm lower_pass in
    let _ = PassManager.run pm mlir_modul in
    ()
  in
  IR.Operation.dump @@ IR.Module.operation mlir_modul
