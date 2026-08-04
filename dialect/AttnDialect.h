//===- AttnDialect.h - Attention dialect ------------------------*- C++ -*-===//
//
// Public header for the `attn` dialect. Include this to build or inspect
// attention IR.
//
//===----------------------------------------------------------------------===//

#ifndef ATTN_DIALECT_H
#define ATTN_DIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Dialect class declaration, generated from AttnDialect.td.
#include "AttnDialect.h.inc"

// Op class declarations, generated from AttnDialect.td.
#define GET_OP_CLASSES
#include "AttnOps.h.inc"

#endif // ATTN_DIALECT_H
