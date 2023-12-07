open Base
module C = Configurator.V1

let () =
  C.main ~name:"mlir" (fun t ->
    let llvm_config flag =
      (try C.Process.(run_capture_exn t "llvm-config-16" [ flag ]) with
       | _ -> failwith "llvm-config not found")
      |> String.strip
    in
    let llvm_libdir = llvm_config "--libdir" in
    let llvm_includedir = llvm_config "--includedir" in
    let libs =
      [ Printf.sprintf "-L%s" llvm_libdir
      ; "-lMLIRCAPIIR"
      ; "-lMLIRCAPIRegisterEverything"
      ; "-lMLIRCAPITransforms"
      ; "-lMLIRCAPIConversion"
      ; "-lMLIRCAPIExecutionEngine"
      ; "-lMLIRCAPILLVM"
      ; "-lLLVM"
      ; "-lMLIR"
      ; "-lstdc++"
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
      ; Printf.(sprintf "-I%s" llvm_includedir)
      ]
    in
    C.Flags.write_sexp "c_flags.sexp" cflags;
    C.Flags.write_sexp "c_library_flags.sexp" libs;
    C.Flags.write_lines "c_flags" cflags;
    C.Flags.write_lines "c_library_flags" libs)
