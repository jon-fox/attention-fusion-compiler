# TableGen the dialect and op classes from AttnDialect.td.
set(LLVM_TARGET_DEFINITIONS AttnDialect.td)
mlir_tablegen(AttnDialect.h.inc -gen-dialect-decls -dialect=attn)
mlir_tablegen(AttnDialect.cpp.inc -gen-dialect-defs -dialect=attn)
mlir_tablegen(AttnOps.h.inc -gen-op-decls)
mlir_tablegen(AttnOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(AttnDialectIncGen)

add_mlir_dialect_library(MLIRAttnDialect
  AttnDialect.cpp
  AttnOps.cpp

  DEPENDS
  AttnDialectIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
  MLIRSupport
)

# The generated .inc files land in the build tree next to this directory.
target_include_directories(MLIRAttnDialect
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)
