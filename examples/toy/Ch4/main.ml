open Parser4.Ast

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
        ; VarDecl ("d", [||], Call ("multiply_transpose", [ Var "a"; Var "b" ]))
        ; Print (Var "d")
        ; Return None
        ] )
  ]


open Mlir

let () =
  let mlir_modul = Mlir_gen.mlirgen modul in
  Inliner.inline_calls_in_main mlir_modul;
  Shape_inference.infer_shapes mlir_modul;
  IR.Operation.dump (IR.Module.operation mlir_modul)
