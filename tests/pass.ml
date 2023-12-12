(* Recreating CAPI test: pass.c *)
open Mlir
open IR

let register_and_load_upstream_dialects ctx =
  let reg = DialectRegistry.create () in
  RegisterEverything.dialects reg;
  Context.append_dialect_registry ctx reg;
  DialectRegistry.destroy reg;
  let load name =
    let _ = IR.Context.get_or_load_dialect ctx name in
    ()
  in
  load "func";
  load "arith"


(* test run pass on module *)
let%expect_test _ =
  with_context (fun ctx ->
    register_and_load_upstream_dialects ctx;
    let m =
      {|
    func.func @foo(%arg0 : i32) -> i32 {
        %res = arith.addi %arg0, %arg0 : i32
        return %res : i32 
    }
    |}
      |> Module.parse ctx
    in
    assert (Module.(not (is_null m)));
    (* Run the print-op-stats pass on the top-level module *)
    let pm = PassManager.create ctx in
    let print_op_stats_pass = Transforms.PrintOpStats.create () in
    PassManager.add_owned_pass pm print_op_stats_pass;
    let success = PassManager.run pm m in
    assert success;
    PassManager.destroy pm;
    Module.destroy m);
  [%expect
    {|
      Operations encountered:
      -----------------------
          arith.addi   , 1
        builtin.module , 1
           func.func   , 1
           func.return , 1
      |}]


(* test run pass on nested module *)
let%expect_test _ =
  with_context (fun ctx ->
    register_and_load_upstream_dialects ctx;
    let m =
      Module.parse
        ctx
        {|
      func.func @foo(%arg0 : i32) -> i32 {
        %res = arith.addi %arg0, %arg0 :i32
        return %res : i32
      }
      module {
        func.func @bar(%arg0 : f32) -> f32 {
          %res = arith.addf %arg0, %arg0 : f32
          return %res : f32
        }
      }
      |}
    in
    assert (not Module.(is_null m));
    let pm = PassManager.create ctx in
    let nested_func_pm = PassManager.nested_under pm "func.func" in
    let print_op_stat_pass = Transforms.PrintOpStats.create () in
    OpPassManager.add_owned_pass nested_func_pm print_op_stat_pass;
    let success = PassManager.run pm m in
    assert success;
    PassManager.destroy pm;
    let pm = PassManager.create ctx in
    let nested_module_pm = PassManager.nested_under pm "builtin.module" in
    let nested_func_pm = OpPassManager.nested_under nested_module_pm "func.func" in
    let print_op_stat_pass = Transforms.PrintOpStats.create () in
    OpPassManager.add_owned_pass nested_func_pm print_op_stat_pass;
    let success = PassManager.run pm m in
    assert success;
    PassManager.destroy pm;
    Module.destroy m);
  [%expect
    {|
      Operations encountered:
      -----------------------
        arith.addi   , 1
         func.func   , 1
         func.return , 1
      Operations encountered:
      -----------------------
        arith.addf   , 1
         func.func   , 1
         func.return , 1
      |}]


(* test print pass pipeline *)
let%expect_test _ =
  with_context (fun ctx ->
    let pm = PassManager.create_on_op ctx "any" in
    let nested_module_pm = PassManager.nested_under pm "builtin.module" in
    let nested_func_pm = OpPassManager.nested_under nested_module_pm "func.func" in
    let print_op_stat_pass = Transforms.PrintOpStats.create () in
    OpPassManager.add_owned_pass nested_func_pm print_op_stat_pass;
    Printf.printf "Top-level: %!";
    OpPassManager.print_pass_pipeline
      ~callback:print_string
      PassManager.(to_op_pass_manager pm);
    Printf.printf "\n%!";
    Printf.printf "Nested Module: %!";
    OpPassManager.print_pass_pipeline ~callback:print_string nested_module_pm;
    Printf.printf "\n%!";
    Printf.printf "Nested Module>Func: %!";
    OpPassManager.print_pass_pipeline ~callback:print_string nested_func_pm;
    Printf.printf "\n%!";
    PassManager.destroy pm);
  [%expect
    {|
      Top-level: any(builtin.module(func.func(print-op-stats{json=false})))
      Nested Module: builtin.module(func.func(print-op-stats{json=false}))
      Nested Module>Func: func.func(print-op-stats{json=false})
      |}]
