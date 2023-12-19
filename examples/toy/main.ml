open Parse
open Mlir

type action =
  | None
  | DumpAST
  | DumpMLIR
  | DumpMLIRAffine
  | DumpMLIRLLVM
  | RunJIT

type config =
  { mutable action : action
  ; mutable enable_opt : bool
  ; mutable filename : string option
  }

let config = { action = None; enable_opt = false; filename = None }

let set_action = function
  | "ast" -> config.action <- DumpAST
  | "mlir" -> config.action <- DumpMLIR
  | "mlir-affine" -> config.action <- DumpMLIRAffine
  | "mlir-llvm" -> config.action <- DumpMLIRLLVM
  | "jit" -> config.action <- RunJIT
  | _ -> Printf.eprintf "wrong -emit arg"


let () =
  let emit_doc = "select the kind of output desired" in
  let args =
    [ "-emit", Arg.String set_action, emit_doc
    ; "-opt", Arg.Unit (fun () -> config.enable_opt <- true), "enable optimizations"
    ; "-f", Arg.String (fun s -> config.filename <- Some s), "path to toy file"
    ]
  in
  Arg.parse
    args
    (fun _ -> Printf.eprintf "wrong arg")
    (Format.sprintf
       "Use -emit to %s. Available options: [ast], [mlir], [mlir-affine], [mlir-llvm], \
        [jit]"
       emit_doc);
  match config.action with
  | None -> ()
  | _ ->
    (match config.filename with
     | None -> Printf.eprintf "empty filename"
     | Some filename ->
       let toy_str = In_channel.(with_open_text filename input_all) in
       let modul = Parser.parse toy_str in
       (match config.action with
        | DumpAST -> print_endline @@ Ast.show_modul modul
        | _ ->
          let toy_dhandle = IR.DialectHandle.get "toy" in
          let () = IR.DialectHandle.register toy_dhandle IR.Context.global_ctx in
          let _ = IR.Context.get_or_load_dialect IR.Context.global_ctx "toy" in
          (match Mlir_gen.mlirgen modul with
           | Result.Ok modul ->
             (match config with
              | { action = DumpMLIR; enable_opt = false; _ } ->
                IR.Operation.dump @@ IR.Module.operation modul
              | _ ->
                let registry = IR.DialectRegistry.create () in
                let () = RegisterEverything.dialects registry in
                let () = RegisterEverything.passes () in
                let () =
                  IR.Context.append_dialect_registry IR.Context.global_ctx registry
                in
                let _ = IR.Context.get_or_load_dialect IR.Context.global_ctx "affine" in
                let _ = IR.Context.get_or_load_dialect IR.Context.global_ctx "memref" in
                let _ = IR.Context.get_or_load_dialect IR.Context.global_ctx "arith" in
                let _ = IR.Context.get_or_load_dialect IR.Context.global_ctx "func" in
                let _ = IR.Context.get_or_load_dialect IR.Context.global_ctx "llvm" in
                let is_to_affine =
                  match config.action with
                  | DumpMLIRAffine | DumpMLIRLLVM | RunJIT -> true
                  | _ -> false
                in
                let is_to_llvm =
                  match config.action with
                  | DumpMLIRLLVM | RunJIT -> true
                  | _ -> false
                in
                let pm = PassManager.create IR.Context.global_ctx in
                let () =
                  if config.enable_opt || is_to_affine
                  then (
                    let inline_pass = Inliner.pass () in
                    let () = PassManager.add_owned_pass pm inline_pass in
                    let toy_func_op_pm = PassManager.nested_under pm "toy.func" in
                    let () =
                      OpPassManager.add_owned_pass
                        toy_func_op_pm
                        (Shape_inference.infer_shapes_pass ())
                    in
                    OpPassManager.add_pipeline
                      toy_func_op_pm
                      "canonicalize,cse"
                      ~callback:print_endline)
                in
                let () =
                  if is_to_affine
                  then (
                    let () =
                      OpPassManager.add_owned_pass
                        (PassManager.to_op_pass_manager pm)
                        (Lower_to_affine.lower_to_affine_pass ())
                    in
                    let func_op_pm = PassManager.nested_under pm "func.func" in
                    let () =
                      OpPassManager.add_pipeline
                        func_op_pm
                        "canonicalize,cse"
                        ~callback:print_endline
                    in
                    if config.enable_opt
                    then
                      OpPassManager.add_pipeline
                        func_op_pm
                        "affine-loop-fusion,affine-scalrep"
                        ~callback:print_endline)
                in
                let () =
                  if is_to_llvm
                  then
                    PassManager.add_owned_pass
                      pm
                      (PassManager.get "createToyToLLVMLoweringPass")
                in
                let () =
                  if not (PassManager.run pm modul) then print_endline "Pass fails"
                in
                (match config.action with
                 | RunJIT ->
                   let () = RegisterEverything.llvm_translations IR.Context.global_ctx in
                   let opt_lvl = if config.enable_opt then 3 else 0 in
                   let jit = ExecutionEngine.create modul opt_lvl [] false in
                   if not @@ ExecutionEngine.invoke_packed jit "main"
                   then print_endline "JIT fails"
                 | _ -> IR.Operation.dump @@ IR.Module.operation modul))
           | Result.Error msg -> Printf.eprintf "Mlir_gen error: %s" msg)))
