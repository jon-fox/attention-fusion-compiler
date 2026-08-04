//===- AttnOps.cpp - Attention op builders and verifiers --------*- C++ -*-===//
//
// Custom builders (which infer result types so callers don't have to spell
// them out) and verifiers for the `attn` ops.
//
//===----------------------------------------------------------------------===//

#include "AttnDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::attn;

// Op definitions, generated from AttnDialect.td.
#define GET_OP_CLASSES
#include "AttnOps.cpp.inc"

//===----------------------------------------------------------------------===//
// attn.matmul
//===----------------------------------------------------------------------===//

void MatMulOp::build(OpBuilder &builder, OperationState &state, Value lhs,
                     Value rhs) {
  auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
  auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());

  // Result is MxN taken from lhs rows and rhs columns. If the operands aren't
  // the rank-2 tensors this op requires, fall back to the lhs type and let the
  // verifier report it — a builder shouldn't assert on malformed input.
  Type resultType = lhs.getType();
  if (lhsType && rhsType && lhsType.getRank() == 2 && rhsType.getRank() == 2)
    resultType =
        RankedTensorType::get({lhsType.getDimSize(0), rhsType.getDimSize(1)},
                              lhsType.getElementType());

  build(builder, state, resultType, lhs, rhs);
}

LogicalResult MatMulOp::verify() {
  // Operand type constraints have already run, so these casts are safe.
  auto lhsType = cast<RankedTensorType>(getLhs().getType());
  auto rhsType = cast<RankedTensorType>(getRhs().getType());
  auto resType = cast<RankedTensorType>(getResult().getType());

  if (lhsType.getRank() != 2 || rhsType.getRank() != 2)
    return emitOpError("expects rank-2 operands, got ranks ")
           << lhsType.getRank() << " and " << rhsType.getRank();

  if (resType.getRank() != 2)
    return emitOpError("expects a rank-2 result, got rank ")
           << resType.getRank();

  if (lhsType.getElementType() != rhsType.getElementType() ||
      lhsType.getElementType() != resType.getElementType())
    return emitOpError("requires one element type throughout, got ")
           << lhsType.getElementType() << ", " << rhsType.getElementType()
           << " and " << resType.getElementType();

  // Only statically known dimensions are checked; dynamic ones are the
  // lowering's problem, not the verifier's.
  int64_t lhsK = lhsType.getDimSize(1);
  int64_t rhsK = rhsType.getDimSize(0);
  if (!ShapedType::isDynamic(lhsK) && !ShapedType::isDynamic(rhsK) &&
      lhsK != rhsK)
    return emitOpError("contraction dimension mismatch: lhs has ")
           << lhsK << " columns but rhs has " << rhsK << " rows";

  int64_t m = lhsType.getDimSize(0);
  int64_t resM = resType.getDimSize(0);
  if (!ShapedType::isDynamic(m) && !ShapedType::isDynamic(resM) && m != resM)
    return emitOpError("result has ")
           << resM << " rows but lhs has " << m;

  int64_t n = rhsType.getDimSize(1);
  int64_t resN = resType.getDimSize(1);
  if (!ShapedType::isDynamic(n) && !ShapedType::isDynamic(resN) && n != resN)
    return emitOpError("result has ")
           << resN << " columns but rhs has " << n;

  return success();
}

//===----------------------------------------------------------------------===//
// attn.scale
//===----------------------------------------------------------------------===//

void ScaleOp::build(OpBuilder &builder, OperationState &state, Value input,
                    float scale) {
  build(builder, state, input.getType(), input,
        builder.getF32FloatAttr(scale));
}

//===----------------------------------------------------------------------===//
// attn.softmax
//===----------------------------------------------------------------------===//

LogicalResult SoftmaxOp::verify() {
  auto inputType = cast<RankedTensorType>(getInput().getType());

  // Read the attribute as signed; the generated accessor widens to uint64_t,
  // which would silently turn a negative axis into a huge positive one.
  int64_t axis = getAxisAttr().getInt();
  int64_t rank = inputType.getRank();

  if (axis < 0 || axis >= rank)
    return emitOpError("axis ")
           << axis << " is out of range for a rank-" << rank << " input";

  return success();
}
