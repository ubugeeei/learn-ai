# 26 — Symmetric int8 weight quantization

## What you will build

Quantize each `Double` matrix row to signed int8 plus one scale, compute matvec
without materializing full-precision weights, and measure maximum, mean, and
root-mean-square reconstruction error. Source:
`src/main/scala/learnai/quantization/Int8Quantization.scala`.

## Why quantize?

Lower-bit weights can reduce:

- storage and download size;
- memory capacity requirements;
- memory bandwidth;
- arithmetic cost on supported hardware.

They also add rounding error, scaling metadata, and kernel requirements. An
eight-times-smaller payload does not guarantee eight-times-lower latency.
Measure the actual bottleneck.

## Symmetric int8

For one row, let \(a=\max_i|w_i|\):

\[
s=\frac{a}{127}
\]

\[
q_i=\operatorname{clamp}
\left(\operatorname{round}(w_i/s),-127,127\right)
\]

Reconstruct with:

\[
\hat w_i=sq_i
\]

Code `-128` remains unused so positive and negative ranges are symmetric. The
zero point is zero. An all-zero row uses scale `1`; all codes remain zero.

## Per-tensor versus per-row scale

A single scale for the whole matrix has little metadata but can waste
resolution when rows have very different magnitudes. Per-row quantization uses:

```text
weights: [outputChannels,inputChannels]
scales:  [outputChannels]
```

It adapts to each output channel at the cost of scale metadata. Production
systems also use per-group scaling and smaller scale dtypes.

## Error bound

Nearest rounding without clipping has code error at most `0.5`, so approximate
absolute reconstruction error is:

\[
|w_i-\hat w_i|\leq s/2
\]

Scale comes from the row maximum, so source weights fit the representable
range. Clamping remains defensive against finite arithmetic effects.

## Quantized matrix-vector multiplication

\[
y_r=\sum_c w_{rc}x_c
\approx s_r\sum_c q_{rc}x_c
\]

The row scale can be applied after accumulation. The teaching implementation
keeps input and accumulator as `Double` so the experiment isolates weight
quantization error.

Production choices include int8 activation with int32 accumulation, weight-only
int4 with fp16 compute, and hardware-specific packed kernels.

## Error metrics

- maximum absolute error: worst element;
- mean absolute error: average magnitude;
- RMSE: emphasizes large errors through squaring.

Small weight error does not guarantee preserved model quality. Compare
activations, logits, perplexity, and task evaluation. Outlier channels and rare
token logits may be especially sensitive.

## Memory estimate

For matrix `R x C`:

- Double payload: \(8RC\) bytes;
- per-row int8 payload: \(RC+8R\) bytes in this implementation.

Object headers, alignment, packing, scale dtype, and kernel workspace also
matter in real measurements.

## PTQ and QAT

This chapter implements post-training quantization:

- **PTQ** converts an already trained model and may use calibration data;
- **QAT** simulates quantization during training so the model adapts.

Lower bit widths are more likely to require fine-tuning, but the result depends
on model, data, and architecture.

## Run it

```console
$ nix develop -c sbt 'runMain learnai.quantization.runInt8QuantizationLab'
```

The experiment reports payload bytes and reconstruction error. Performance
claims require a warmed repeated benchmark.

## Implementation walkthrough

`QuantizedInt8Matrix.quantize` processes one matrix row at a time. It finds the
largest absolute value and chooses

\[
s_r = \max_j |w_{rj}| / 127
\]

For an all-zero row it uses a valid non-zero scale so dequantization and matvec
never divide by zero. Each weight is divided by the row scale, rounded to the
nearest integer, and clamped to `[-127,127]`. The implementation avoids `-128`
to keep symmetric positive/negative range.

For row `[-2,-1,0,1,2]`, scale is `2/127`. Codes are approximately
`[-127,-64,0,64,127]`; reconstructed `64*(2/127)` is about `1.0079`. The error
comes from rounding to the nearest grid point.

The quantized object owns byte-like integer payload and one `Double` scale per
row. `storageBytes` counts payload plus scale data explicitly. `matvec` does not
materialize a full dequantized matrix: it multiplies each code by the row scale
inside the dot product. This saves storage but the Scala loop is not a packed
SIMD int8 kernel.

`QuantizationMetrics.compare` checks equal shape, then reports maximum absolute
error, mean absolute error, and root mean squared error. These describe weight
reconstruction, not language-model quality.

## Reading the tests

All-zero rows define the scale boundary. Row extrema must map to symmetric
limits. The half-scale error bound comes from nearest-grid rounding and is an
independent mathematical oracle. Quantized matvec is compared with `MatrixD`
reference. Exact reconstruction gives zero metrics. Storage tests choose a row
wide enough that scale metadata does not erase payload savings.

## Debugging checklist

1. Print row maximum, scale, raw quotient, rounded code, and reconstruction.
2. Handle the zero row before division.
3. Verify clamping and symmetric range.
4. Separate weight error, output error, task quality, storage, and latency.
5. Do not claim faster inference without a packed kernel benchmark on target
   hardware.

## Exercises

1. Quantize row `[-2,-1,0,1,2]` by hand.
2. Implement per-tensor scaling and compare mixed-magnitude rows.
3. Store scales as `Float` and compare bytes and error.
4. Measure matvec output cosine similarity.
5. Quantize MiniGPT 2D weights and compare fixed-prompt logits.
6. Design packed int4 groups and bit operations.

## Completion criteria

- Compute scale, code, and reconstructed value.
- Explain why symmetric zero point is zero.
- Compare per-row and per-tensor metadata/error.
- Explain why weight error and task quality both matter.
- Distinguish storage reduction from latency improvement.
- `Int8QuantizationSuite` passes.

## Primary sources

- [GPTQ](https://arxiv.org/abs/2210.17323)
- [AWQ](https://arxiv.org/abs/2306.00978)
- [Course reading map and critical summaries](../09-papers/40-primary-reading-map.md)
