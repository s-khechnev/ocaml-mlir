open Parser1.Ast

let modul =
  [ Function
      ( Prototype ("multiply_transpose", [ "a"; "b" ])
      , [ Return
            (Some
               (BinOp
                  ('*', Call ("transpose", [ Var "a" ]), Call ("transpose", [ Var "b" ]))))
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
        ] )
  ]


open Mlir
open IR

let get =
  let open Ctypes in
  Foreign.foreign
    "mlirGetDialectHandle__toy__"
    (void @-> returning Stubs.Typs.DialectHandle.t)


let () =
  let ctx = Context.create () in
  let dhandle = get () in
  let () = DialectHandle.register dhandle ctx in
  let dialect = Context.get_or_load_dialect ctx "toy" in
  print_endline @@ Dialect.namespace dialect
