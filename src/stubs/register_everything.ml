open Ctypes

module Bindings (F : FOREIGN) = struct
  open F

  (* Appends all upstream dialects and extensions to the dialect registry. *)
  let dialects =
    foreign "mlirRegisterAllDialects" (Typs.DialectRegistry.t @-> returning void)


  (* Register all translations to LLVM IR for dialects that can support it. *)
  let llvm_translations =
    foreign "mlirRegisterAllLLVMTranslations" (Typs.Context.t @-> returning void)


  (* Register all compiler passes of MLIR. *)
  let passes = foreign "mlirRegisterAllPasses" (void @-> returning void)
end
