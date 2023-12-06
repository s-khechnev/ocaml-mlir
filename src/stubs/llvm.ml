open Ctypes
open Typs

module Bindings (F : FOREIGN) = struct
  open F

  let get = foreign "mlirGetDialectHandle__llvm__" (void @-> returning DialectHandle.t)

  (** Creates an llvm.ptr type. *)
  let pointer = foreign "mlirLLVMPointerTypeGet" (Type.t @-> uint @-> returning Type.t)

  (** Creates an llmv.void type. *)
  let void_t = foreign "mlirLLVMVoidTypeGet" (Context.t @-> returning Type.t)

  (** Creates an llvm.array type. *)
  let arr = foreign "mlirLLVMArrayTypeGet" (Type.t @-> uint @-> returning Type.t)

  (** Creates an llvm.func type. *)
  let func =
    foreign
      "mlirLLVMFunctionTypeGet"
      (Type.t @-> intptr_t @-> ptr Type.t @-> bool @-> returning Type.t)


  (** Creates an LLVM literal (unnamed) struct type. *)
  let literal_struct =
    foreign
      "mlirLLVMStructTypeLiteralGet"
      (Context.t @-> intptr_t @-> ptr Type.t @-> bool @-> returning Type.t)
end
