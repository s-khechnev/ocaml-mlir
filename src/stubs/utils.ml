open Ctypes

module Make
    (S : sig
       val s : string
     end)
    (K : sig
       val k : string
     end)
    (F : FOREIGN) =
struct
  open F

  let create =
    foreign Printf.(sprintf "mlirCreate%s%s" K.k S.s) (void @-> returning Typs.Pass.t)


  let register =
    foreign Printf.(sprintf "mlirRegister%s%s" K.k S.s) (void @-> returning void)
end

module MakeTransform
    (S : sig
       val s : string
     end)
    (F : FOREIGN) =
  Make
    (S)
    (struct
      let k = "Transforms"
    end)
    (F)

module MakeConversion
    (S : sig
       val s : string
     end)
    (F : FOREIGN) =
  Make
    (S)
    (struct
      let k = "Conversion"
    end)
    (F)
