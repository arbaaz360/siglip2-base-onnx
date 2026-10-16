# SigLIP 2 base, as ONNX

Google's [`google/siglip2-base-patch16-256`](https://huggingface.co/google/siglip2-base-patch16-256)
exported to ONNX as two separate models: one turns a picture into a vector, the
other turns a sentence into a vector in the same space, so the cosine of the two
says how well the words describe the picture. Built for on-device photo search;
nothing here needs a GPU or a server.

The files are attached to releases. This repository holds no code.

| Release | Download | What changed |
|---|---|---|
| [`models-siglip2-base-v3`](https://github.com/arbaaz360/siglip2-base-onnx/releases/tag/models-siglip2-base-v3) (current) | 559 MB | the vision tower runs on Core ML (macOS): explicit convolution padding, float32 arithmetic; text tower and tokenizer as v2 |
| [`models-siglip2-base-v2`](https://github.com/arbaaz360/siglip2-base-onnx/releases/tag/models-siglip2-base-v2) | 559 MB | the text tower's token table is int8; everything else as v1 |
| [`models-siglip2-base-v1`](https://github.com/arbaaz360/siglip2-base-onnx/releases/tag/models-siglip2-base-v1) | 755 MB | first export; kept for the builds that point at it |

`text.onnx` is byte-identical in v2 and v3, `tokenizer.model` in all three.
Image vectors from all three agree to cosine ≥ 0.99999, so an index built with
one works with the others.

## Files (v3)

| File | Bytes | SHA-256 | What |
|---|---|---|---|
| `vision.onnx` | 186,151,511 | `25c1825ef9c20d4f3ab5d20910afacf96f40496827264c28385f45d82ce00e60` | `pixel_values [batch,3,256,256] float32` → `image_embeds [batch,768]`, L2-normalised |
| `text.onnx` | 368,714,934 | `3cf1137e1c9b9a73dda692d66dd0b2b95988b85b6bc3d4347c87f3b2ee485039` | `input_ids [batch,64] int64` → `text_embeds [batch,768]`, L2-normalised |
| `tokenizer.model` | 4,241,003 | `61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2` | the Gemma SentencePiece tokenizer, unchanged from the checkpoint |
| `manifest.json` | — | — | the list above as JSON, with the download URLs and the input shapes |

Earlier files: v1 and v2's `vision.onnx` was 186,099,025 bytes, SHA-256
`ae478837056f3038d6793d4eb6450f4eacc8fc3b38711a4cd1d899af3f4747c7`; v1's
`text.onnx` was 564,810,160 bytes, SHA-256
`958aab2836acfcb43b431c5b8215a6b5b778b7998f81cdbd6ea5e0c7799355b0`.

## Using them

| | |
|---|---|
| Image input | resize to 256×256 without cropping, RGB, `(x / 255 − 0.5) / 0.5`, NCHW |
| Text input | Gemma SentencePiece ids; no BOS; EOS (id 1) appended; right-padded with 0 to 64; content truncated so the EOS fits; case kept as typed |
| Output | 768 floats, already unit length: the dot product is the cosine |
| Weights | float16 in the file, except the text tower's token table (int8, one float16 scale per row, dequantised after the lookup). From v3 the vision tower casts its weights to float32 and computes in float32; the text tower computes in float16. Inputs and outputs float32; opset 17, standard operators only |
| Batch | the first dimension of both inputs is named `batch` and is free. Core ML needs it fixed: with ONNX Runtime, `SessionOptions.AddFreeDimensionOverrideByName("batch", n)` and pad short batches |
| Fidelity | cosine against the PyTorch model on ten images and ten sentences: images 1.00000 (v3; v1 and v2 ≥ 0.99999), text ≥ 0.99928 (v1 text: ≥ 0.99999) |

SigLIP is trained with a sigmoid loss, so raw cosines are small (about 0.05 to
0.2 for a good match); rank by them rather than reading them as probabilities.

### Why v3: Core ML

v1 and v2's vision tower could not run on Core ML through ONNX Runtime's CoreML
provider, for two reasons found on GitHub's macOS 26 runner:

1. Core ML refused to load it: the patch-embedding convolution said
   `auto_pad=VALID`, which the provider writes as an ML Program `conv` without
   its `pad` parameter ("Required param 'pad' is missing"). v3 writes
   `pads=[0,0,0,0]` instead; the arithmetic is the same.
2. Core ML runs a float16 graph in float16 arithmetic throughout. This tower
   has a few residual-stream values near 700 from layer 6 on, and in float16
   the image vectors drifted to cosine 0.983–0.994 against ONNX Runtime's CPU
   provider. Computing in float32 fixed it; computing only the MLP, only the
   attention or only the layer norms in float32 did not:

| Vision tower on Core ML, against the CPU, 16 pictures | Lowest cosine |
|---|---|
| float16 arithmetic (v2 with the padding fixed) | 0.98341 |
| layer norms in float32 | 0.98338 |
| attention in float32 | 0.98353 |
| MLP in float32 | 0.99523 |
| **everything in float32 (v3)** | **1.00000** |

What float32 arithmetic costs elsewhere, batch 8, 1600×1200 sources:

| | v2 | v3 |
|---|---|---|
| ONNX Runtime CPU, 8 threads (Core Ultra 7 265K) | 10.3–10.5 img/s | 11.0–11.2 img/s |
| ONNX Runtime DirectML (RTX 5060 Ti) | 168–174 img/s | 100–101 img/s |

The CPU provider already widened float16 to float32 when it loaded v2, so it
loses nothing; a GPU with fast float16 does.

### What the int8 table costs (v2)

The text tower is mostly its vocabulary: 256,000 rows of 768, 393 MB of the
565 MB in float16. v2 stores that one table as int8 with a scale per row
(`max |row| / 127`), and the graph rebuilds only the 64 rows a sentence uses:
`Gather(int8) → Cast → Mul(Gather(scale))`.

| | v1 | v2 |
|---|---|---|
| `text.onnx` | 565 MB | 369 MB |
| Text vector vs PyTorch, worst of ten sentences | 0.99999 | 0.99928 |
| Nine ground-truth photo queries, ranked against the image vectors a v1 app stored: same first hit as v1 | | 9 of 9, top five identical in order |
| 25 queries over 236 pictures: same first hit as v1 | | 23 of 25 (the two changes were ties within 0.0004; PyTorch float32: 24 of 25) |
| Text tower, CPU memory after load (ONNX Runtime) | 1,144 MB | 555 MB |

## Licence and changes

The model and tokenizer are © Google, released under the Apache License 2.0
(see [`LICENSE`](LICENSE) and the
[model card](https://huggingface.co/google/siglip2-base-patch16-256)).

Changes made here, as section 4(b) of the licence asks:

- **v1:** the checkpoint was converted to ONNX; the vision and text towers
  were split into separate graphs; weights were cast to float16; L2
  normalisation was added to the end of each graph. The weights themselves
  were not retrained or altered beyond the cast.
- **v2:** as v1, and the text tower's token embedding table
  (`text_model.embeddings.token_embedding.weight`, 256,000 × 768) was
  quantised to int8 with one float16 scale per row, with the dequantisation
  added to the graph. This changes those weights slightly (each row's cosine
  with the original is at least 0.99957). Nothing was retrained; the vision
  tower and the tokenizer are unchanged from v1.
- **v3:** as v2, and in the vision tower the patch-embedding convolution's
  padding is written as explicit zero `pads` instead of `auto_pad=VALID`
  (same arithmetic), and Cast nodes make the vision graph compute in float32
  from its float16 weights. The weights are the same float16 values as v1
  and v2; nothing was retrained. The text tower and the tokenizer are
  byte-identical to v2.

This repository is not affiliated with or endorsed by Google.
