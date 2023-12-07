open Ctypes
module Typs = Typs

module Bindings (F : FOREIGN) = struct
  module Dialect = struct
    module LLVM = Llvm.Bindings (F)
  end

  module AffineExpr = Affine_expr.Bindings (F)
  module AffineMap = Affine_map.Bindings (F)
  module Diagnostics = Diagnostics.Bindings (F)
  module IR = Ir.Bindings (F)
  module Pass = Pass.Bindings (F)
  module RegisterEverything = Register_everything.Bindings (F)
  module BuiltinAttributes = Builtin_attributes.Bindings (F)
  module BuiltinTypes = Builtin_types.Bindings (F)
  module Transforms = Transforms.Bindings (F)
  module Conversion = Conversion.Bindings (F)
  module Support = Support.Bindings (F)
end
