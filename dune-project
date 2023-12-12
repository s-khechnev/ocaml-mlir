(lang dune 2.7)

(generate_opam_files true)

(source
 (github s-khechnev/ocaml-mlir))

(license MIT)

(authors "Ta-Chu Kao")

(name mlir)

(package
 (name mlir)
 (synopsis "Bindings to LLVM MLIR")
 (description "OCaml bindings to LLVM MLIR")
 (depends
  (ocaml
   (>= 4.14))
  ctypes
  ctypes-foreign
  stdune
  ppx_inline_test
  ppx_expect
  bisect_ppx
  (base :with-test)
  (ocamlformat
   (>= 0.23.0))))
