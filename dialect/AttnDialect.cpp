//===- AttnDialect.cpp - Attention dialect registration ---------*- C++ -*-===//

#include "AttnDialect.h"

// Dialect definitions, generated from AttnDialect.td.
#include "AttnDialect.cpp.inc"

using namespace mlir;
using namespace mlir::attn;

void AttnDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "AttnOps.cpp.inc"
      >();
}
