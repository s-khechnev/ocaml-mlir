#include "Dialects.h"

#include "rgn/Dialect.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Rgn, rgn,
                                      mlir::rgn::RgnDialect)
