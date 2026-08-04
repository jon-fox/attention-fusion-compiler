//===- build_graph.cpp - Build and print an attention graph -----*- C++ -*-===//
//
// Constructs the scores half of attention in the `attn` dialect:
//
//     matmul -> scale -> softmax
//
// then verifies the module and prints it. Exits non-zero if verification
// fails, so this doubles as a smoke test for the dialect.
//
//===----------------------------------------------------------------------===//

#include "AttnDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"

#include <cmath>

using namespace mlir;

int main() {
  MLIRContext context;
  context.getOrLoadDialect<attn::AttnDialect>();
  context.getOrLoadDialect<func::FuncDialect>();

  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();

  // A single attention head: 128 query positions, 128 key positions, head
  // dimension 64. K arrives pre-transposed, since attn.matmul only multiplies.
  const int64_t seqLen = 128;
  const int64_t headDim = 64;
  Type f32 = builder.getF32Type();
  auto queryType = RankedTensorType::get({seqLen, headDim}, f32);
  auto keyTType = RankedTensorType::get({headDim, seqLen}, f32);
  auto scoresType = RankedTensorType::get({seqLen, seqLen}, f32);

  ModuleOp module = ModuleOp::create(builder, loc);
  builder.setInsertionPointToEnd(module.getBody());

  auto fnType = builder.getFunctionType({queryType, keyTType}, {scoresType});
  auto fn = func::FuncOp::create(builder, loc, "attention_scores", fnType);
  Block &body = *fn.addEntryBlock();
  builder.setInsertionPointToStart(&body);

  Value query = body.getArgument(0);
  Value keyT = body.getArgument(1);

  // matmul -> scale -> softmax. Each builder infers its own result type.
  Value scores = attn::MatMulOp::create(builder, loc, query, keyT);
  Value scaled = attn::ScaleOp::create(
      builder, loc, scores, 1.0f / std::sqrt(static_cast<float>(headDim)));
  Value probs = attn::SoftmaxOp::create(builder, loc, scaled, /*axis=*/1);

  func::ReturnOp::create(builder, loc, probs);

  if (failed(verify(module))) {
    llvm::errs() << "error: module failed verification\n";
    return 1;
  }

  module.print(llvm::outs());
  llvm::outs() << "\n";
  return 0;
}
