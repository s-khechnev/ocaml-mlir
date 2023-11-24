open Mlir.IR

let () =
  let ctx = Context.create () in
  let dhandle = DialectHandle.get "toy" in
  let () = DialectHandle.register dhandle ctx in
  let dialect = Context.get_or_load_dialect ctx "toy" in
  print_endline @@ Dialect.namespace dialect
