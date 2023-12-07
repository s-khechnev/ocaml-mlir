open Ctypes
open Utils

module Bindings (F : FOREIGN) = struct
  open F

  (* Registration for the entire group *)
  let register_passes = foreign "mlirRegisterTransformsPasses" (void @-> returning void)

  module CSE =
    MakeTransform
      (struct
        let s = "CSE"
      end)
      (F)

  module Canonicalizer =
    MakeTransform
      (struct
        let s = "Canonicalizer"
      end)
      (F)

  module ControlFlowSink =
    MakeTransform
      (struct
        let s = "ControlFlowSink"
      end)
      (F)

  module GenerateRuntimeVerification =
    MakeTransform
      (struct
        let s = "GenerateRuntimeVerification"
      end)
      (F)

  module Inliner =
    MakeTransform
      (struct
        let s = "Inliner"
      end)
      (F)

  module LocationSnapshot =
    MakeTransform
      (struct
        let s = "LocationSnapshot"
      end)
      (F)

  module LoopInvariantCodeMotion =
    MakeTransform
      (struct
        let s = "LoopInvariantCodeMotion"
      end)
      (F)

  module PrintOpStats =
    MakeTransform
      (struct
        let s = "PrintOpStats"
      end)
      (F)

  module SCCP =
    MakeTransform
      (struct
        let s = "SCCP"
      end)
      (F)

  module StripDebugInfo =
    MakeTransform
      (struct
        let s = "StripDebugInfo"
      end)
      (F)

  module SymbolDCE =
    MakeTransform
      (struct
        let s = "SymbolDCE"
      end)
      (F)

  module SymbolPrivatize =
    MakeTransform
      (struct
        let s = "SymbolPrivatize"
      end)
      (F)

  module TopologicalSort =
    MakeTransform
      (struct
        let s = "TopologicalSort"
      end)
      (F)

  module ViewOpGraph =
    MakeTransform
      (struct
        let s = "ViewOpGraph"
      end)
      (F)
end
