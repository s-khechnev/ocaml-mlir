open Parser6.Ast

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
  let _ = IR.Context.get_or_load_dialect IR.Context.global_ctx "llvm" in
  let mlir_modul = Mlir_gen.mlirgen modul in
  let () = Inliner.inline_calls_in_main mlir_modul in
  let () =
    (* pipeline: shape_inference, canonicalize, cse, lower_to_affine,
       canonicalize, cse, loop_fusion, affine_scalar_replacement, lower_to_llvm *)
    let () = RegisterEverything.passes () in
    let infer_pass = Shape_inference.infer_shapes_pass () in
    let lower_pass = Lower_to_affine.lower_to_affine_pass () in
    let pm = PassManager.create IR.Context.global_ctx in
    let toy_func_op_pm = PassManager.nested_under pm "toy.func" in
    OpPassManager.add_owned_pass toy_func_op_pm infer_pass;
    OpPassManager.add_pipeline toy_func_op_pm "canonicalize,cse" ~callback:print_endline;
    OpPassManager.add_owned_pass (PassManager.to_op_pass_manager pm) lower_pass;
    let func_op_pm = PassManager.nested_under pm "func.func" in
    OpPassManager.add_pipeline
      func_op_pm
      "canonicalize,cse,affine-loop-fusion,affine-scalrep"
      ~callback:print_endline;
    PassManager.add_owned_pass pm (PassManager.get "createToyToLLVMLoweringPass");
    let _ = PassManager.run pm mlir_modul in
    ()
  in
  IR.Operation.dump @@ IR.Module.operation mlir_modul
