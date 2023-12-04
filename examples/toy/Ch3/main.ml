open Parser3.Ast

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
      , [ VarDecl
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
        ; VarDecl ("c", [||], Call ("multiply_transpose", [ Var "a"; Var "b" ]))
        ; VarDecl ("d", [||], Call ("multiply_transpose", [ Var "b"; Var "a" ]))
        ; Print (Var "d")
        ; Return None
        ] )
  ]


open Mlir

let () =
  let dhandle = IR.DialectHandle.get "toy" in
  let () = IR.DialectHandle.register dhandle IR.Context.global_ctx in
  let _ = IR.Context.get_or_load_dialect IR.Context.global_ctx "toy" in
  let mlir_modul = Mlir_gen.mlirgen modul in
  let () =
    let pm = PassManager.create IR.Context.global_ctx in
    let op_pm = PassManager.nested_under pm "toy.func" in
    let canon_pass = Transforms.Canonicalizer.create () in
    let () = OpPassManager.add_owned_pass op_pm canon_pass in
    let _ = PassManager.run pm mlir_modul in
    ()
  in
  IR.Operation.dump (IR.Module.operation mlir_modul)
