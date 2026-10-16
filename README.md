# SigLIP 2 base, as ONNX

Google's [`google/siglip2-base-patch16-256`](https://huggingface.co/google/siglip2-base-patch16-256)
exported to ONNX as two separate models: one turns a picture into a vector, the
other turns a sentence into a vector in the same space, so the cosine of the two
says how well the words describe the picture. Built for on-device photo search;
nothing here needs a GPU or a server.

The files are attached to releases. This repository holds no code.

| Release | Download | What changed |
|---|---|---|
| [`models-siglip2-base-v2`](https://github.com/arbaaz360/siglip2-base-onnx/releases/tag/models-siglip2-base-v2) (current) | 559 MB | the text tower's token table is int8; everything else as v1 |
| [`models-siglip2-base-v1`](https://github.com/arbaaz360/siglip2-base-onnx/releases/tag/models-siglip2-base-v1) | 755 MB | first export; kept for the builds that point at it |

`vision.onnx` and `tokenizer.model` are byte-identical in both releases, so
image vectors from v1 and v2 are the same and an index built with one works
with the other.

## Files (v2)

| File | Bytes | SHA-256 | What |
|---|---|---|---|
| `vision.onnx` | 186,099,025 | `ae478837056f3038d6793d4eb6450f4eacc8fc3b38711a4cd1d899af3f4747c7` | `pixel_values [N,3,256,256] float32` → `image_embeds [N,768]`, L2-normalised |
| `text.onnx` | 368,714,934 | `3cf1137e1c9b9a73dda692d66dd0b2b95988b85b6bc3d4347c87f3b2ee485039` | `input_ids [N,64] int64` → `text_embeds [N,768]`, L2-normalised |
| `tokenizer.model` | 4,241,003 | `61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2` | the Gemma SentencePiece tokenizer, unchanged from the checkpoint |
| `manifest.json` | — | — | the list above as JSON, with the download URLs |

v1's `text.onnx` was 564,810,160 bytes, SHA-256
`958aab2836acfcb43b431c5b8215a6b5b778b7998f81cdbd6ea5e0c7799355b0`.

## Using them

| | |
|---|---|
| Image input | resize to 256×256 without cropping, RGB, `(x / 255 − 0.5) / 0.5`, NCHW |
| Text input | Gemma SentencePiece ids; no BOS; EOS (id 1) appended; right-padded with 0 to 64; content truncated so the EOS fits; case kept as typed |
| Output | 768 floats, already unit length: the dot product is the cosine |
| Weights | float16 inside the graph, except the text tower's token table (int8, one float16 scale per row, dequantised after the lookup); inputs and outputs float32; opset 17, standard operators only |
| Fidelity | cosine against the PyTorch model on ten images and ten sentences: images ≥ 0.99999, text ≥ 0.99928 (v1 text: ≥ 0.99999) |

SigLIP is trained with a sigmoid loss, so raw cosines are small (about 0.05 to
0.2 for a good match); rank by them rather than reading them as probabilities.

### What the int8 table costs

The text tower is mostly its vocabulary: 256,000 rows of 768, 393 MB of the
565 MB in float16. v2 stores that one table as int8 with a scale per row
(`max |row| / 127`), and the graph rebuilds only the 64 rows a sentence uses:
`Gather(int8) → Cast → Mul(Gather(scale))`.

| | v1 | v2 |
|---|---|---|
| `text.onnx` | 565 MB | 369 MB |
| Text vector vs PyTorch, worst of ten sentences | 0.99999 | 0.99928 |
| Nine photo queries with known answers: same first hit as v1 | | 9 of 9, top five unchanged |
| 25 queries over 228 pictures: same first hit as v1 | | 23 of 25 (PyTorch float32 also 23 of 25) |
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

This repository is not affiliated with or endorsed by Google.
