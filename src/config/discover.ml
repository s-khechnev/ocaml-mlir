open Base
module C = Configurator.V1

let write_flags file list_of_str =
  let data = String.concat list_of_str ~sep:" " in
  Stdio.Out_channel.write_all file ~data


let () =
  C.main ~name:"mlir" (fun t ->
    let llvm_src_root =
      (try C.Process.(run_capture_exn t "llvm-config" [ "--includedir" ]) with
       | _ -> failwith "llvm-config not found")
      |> String.strip
      |> String.chop_suffix_exn ~suffix:"/include"
    in
    let llvm_build =
      (try C.Process.(run_capture_exn t "llvm-config" [ "--prefix" ]) with
       | _ -> failwith "llvm-config not found")
      |> String.strip
    in
    let libs =
      [ Printf.(sprintf "-L%s/../mlir/lib" llvm_src_root)
      ; Printf.(sprintf "-L%s/lib" llvm_src_root)
      ; Printf.(sprintf "-L%s/lib" llvm_build)
      ; Printf.(sprintf "-L%s/tools/mlir/lib" llvm_build) (* ; "-lMLIRPublicAPI" *)
      ; "-lMLIRCAPIIR"
      ]
    in
    let cflags =
      [ "-fPIC"
      ; "-Werror=date-time"
      ; "-Wall"
      ; "-Wextra"
      ; "-Wno-unused-parameter"
      ; "-Wwrite-strings"
      ; "-Wno-missing-field-initializers"
      ; "-Wimplicit-fallthrough"
      ; "-Wno-comment"
      ; "-ffunction-sections"
      ; "-fdata-sections"
      ; "-Iinclude"
      ; Printf.(sprintf "-I%s/../mlir/include" llvm_src_root)
      ; Printf.(sprintf "-I%s/include" llvm_src_root)
      ; Printf.(sprintf "-I%s/include" llvm_build)
      ; Printf.(sprintf "-I%s/tools/mlir/include" llvm_build)
      ; "-I../include"
      ]
    in
    let conf : C.Pkg_config.package_conf = { cflags; libs } in
    C.Flags.write_sexp "c_flags.sexp" conf.cflags;
    C.Flags.write_sexp "c_library_flags.sexp" conf.libs;
    C.Flags.write_lines "c_flags" conf.cflags;
    C.Flags.write_lines "c_library_flags" conf.libs)
