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
