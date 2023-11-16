open Ctypes

module Make
    (S : sig
       val s : string
     end)
    (F : FOREIGN) =
struct
  open F

  let create =
    foreign Printf.(sprintf "mlirCreateTransforms%s" S.s) (void @-> returning Typs.Pass.t)


  let register =
    foreign Printf.(sprintf "mlirRegisterTransforms%s" S.s) (void @-> returning void)
end

module Bindings (F : FOREIGN) = struct
  open F

  (* Registration for the entire group *)
  let register_passes = foreign "mlirRegisterTransformsPasses" (void @-> returning void)

  module CSE =
    Make
      (struct
        let s = "CSE"
      end)
      (F)

  module Canonicalizer =
    Make
      (struct
        let s = "Canonicalizer"
      end)
      (F)

  module ControlFlowSink =
    Make
      (struct
        let s = "ControlFlowSink"
      end)
      (F)

  module GenerateRuntimeVerification =
    Make
      (struct
        let s = "GenerateRuntimeVerification"
      end)
      (F)

  module Inliner =
    Make
      (struct
        let s = "Inliner"
      end)
      (F)

  module LocationSnapshot =
    Make
      (struct
        let s = "LocationSnapshot"
      end)
      (F)

  module LoopInvariantCodeMotion =
    Make
      (struct
        let s = "LoopInvariantCodeMotion"
      end)
      (F)

  module PrintOpStats =
    Make
      (struct
        let s = "PrintOpStats"
      end)
      (F)

  module SCCP =
    Make
      (struct
        let s = "SCCP"
      end)
      (F)

  module StripDebugInfo =
    Make
      (struct
        let s = "StripDebugInfo"
      end)
      (F)

  module SymbolDCE =
    Make
      (struct
        let s = "SymbolDCE"
      end)
      (F)

  module SymbolPrivatize =
    Make
      (struct
        let s = "SymbolPrivatize"
      end)
      (F)

  module TopologicalSort =
    Make
      (struct
        let s = "TopologicalSort"
      end)
      (F)

  module ViewOpGraph =
    Make
      (struct
        let s = "ViewOpGraph"
      end)
      (F)
end
