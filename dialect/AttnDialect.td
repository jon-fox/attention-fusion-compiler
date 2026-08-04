//===- AttnDialect.td - Attention dialect definition -------*- tablegen -*-===//
//
// Defines the `attn` dialect: a small IR for expressing attention as a graph
// of tensor operations, before any fusion or lowering happens.
//
//===----------------------------------------------------------------------===//

#ifndef ATTN_DIALECT_TD
#define ATTN_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

def Attn_Dialect : Dialect {
  let name = "attn";
  let summary = "Operations for expressing attention as a fusable graph";
  let description = [{
    The `attn` dialect represents the pieces of an attention computation as
    separate, side-effect-free tensor operations. Keeping them separate is
    deliberate: fusion passes later match on this graph shape, so the
    unfused form is the input the compiler is meant to optimize.

    Values are ranked tensors of floating point element type. Dynamic
    dimensions are permitted; verifiers only check what is statically known.
  }];
  let cppNamespace = "::mlir::attn";
}

//===----------------------------------------------------------------------===//
// Base classes and shared type constraints
//===----------------------------------------------------------------------===//

class Attn_Op<string mnemonic, list<Trait> traits = []>
    : Op<Attn_Dialect, mnemonic, traits>;

// Every value in the dialect is a ranked tensor of floats.
def Attn_Tensor : RankedTensorOf<[AnyFloat]>;

//===----------------------------------------------------------------------===//
// attn.matmul
//===----------------------------------------------------------------------===//

def Attn_MatMulOp : Attn_Op<"matmul", [Pure]> {
  let summary = "2-D matrix multiply";
  let description = [{
    Computes `lhs * rhs` for rank-2 tensors, contracting the trailing
    dimension of `lhs` against the leading dimension of `rhs`.

    This is the `Q * K^T` and `probs * V` step of attention. Transposition is
    the caller's responsibility; this op only multiplies.

    Example:

    ```mlir
    %0 = attn.matmul %q, %kT : (tensor<128x64xf32>, tensor<64x128xf32>)
                                -> tensor<128x128xf32>
    ```
  }];

  let arguments = (ins Attn_Tensor:$lhs, Attn_Tensor:$rhs);
  let results = (outs Attn_Tensor:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
  }];

  // Infers the MxN result type from the operand shapes.
  let builders = [
    OpBuilder<(ins "::mlir::Value":$lhs, "::mlir::Value":$rhs)>
  ];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// attn.scale
//===----------------------------------------------------------------------===//

def Attn_ScaleOp : Attn_Op<"scale", [Pure, SameOperandsAndResultType]> {
  let summary = "Multiply every element by a constant";
  let description = [{
    Scales `input` elementwise by a compile-time constant. In attention this
    carries the `1/sqrt(d_head)` factor applied to the raw scores.

    The scale is an attribute rather than an operand so that fusion passes can
    fold it into a neighbouring op without tracing a value definition.

    Example:

    ```mlir
    %1 = attn.scale %0, 1.250000e-01 : tensor<128x128xf32>
    ```
  }];

  let arguments = (ins Attn_Tensor:$input, F32Attr:$scale);
  let results = (outs Attn_Tensor:$result);

  let assemblyFormat = [{
    $input `,` $scale attr-dict `:` type($result)
  }];

  let builders = [
    OpBuilder<(ins "::mlir::Value":$input, "float":$scale)>
  ];
}

//===----------------------------------------------------------------------===//
// attn.softmax
//===----------------------------------------------------------------------===//

def Attn_SoftmaxOp : Attn_Op<"softmax", [Pure, SameOperandsAndResultType]> {
  let summary = "Softmax along one axis";
  let description = [{
    Normalizes `input` along `axis` so the values on that axis sum to one.
    For attention scores this is the last axis, over the key positions.

    The op is defined as the numerically stable form (subtract the max along
    the axis before exponentiating); that is a property of the lowering, not
    something the IR spells out.

    Example:

    ```mlir
    %2 = attn.softmax %1 {axis = 1 : i64} : tensor<128x128xf32>
    ```
  }];

  let arguments = (ins Attn_Tensor:$input, I64Attr:$axis);
  let results = (outs Attn_Tensor:$result);

  let assemblyFormat = [{
    $input attr-dict `:` type($result)
  }];

  // No custom builder: SameOperandsAndResultType already makes ODS generate
  // `create(builder, loc, input, axis)` with the result type inferred, which
  // is exactly what a hand-written one would do.

  let hasVerifier = 1;
}

#endif // ATTN_DIALECT_TD
