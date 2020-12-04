open Ctypes
module Typs = Typs

module Bindings (F : FOREIGN) = struct
  module AffineExpr = Affine_expr.Bindings (F)
  include Affine_map.Bindings (F)
  include Diagnostics.Bindings (F)
  include Ir.Bindings (F)
  include Pass.Bindings (F)
  include Registration.Bindings (F)
  include Standard_attributes.Bindings (F)
  module StandardDialect = Standard_dialect.Bindings (F)
  include Standard_types.Bindings (F)
  include Support.Bindings (F)
end
