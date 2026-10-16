# SigLIP 2 base, as ONNX

Google's [`google/siglip2-base-patch16-256`](https://huggingface.co/google/siglip2-base-patch16-256)
exported to ONNX as two separate models: one turns a picture into a vector, the
other turns a sentence into a vector in the same space, so the cosine of the two
says how well the words describe the picture. Built for on-device photo search;
nothing here needs a GPU or a server.

The files are attached to the release
[`models-siglip2-base-v1`](https://github.com/arbaaz360/siglip2-base-onnx/releases/tag/models-siglip2-base-v1).
This repository holds no code.

## Files

| File | Bytes | SHA-256 | What |
|---|---|---|---|
| `vision.onnx` | 186,099,025 | `ae478837056f3038d6793d4eb6450f4eacc8fc3b38711a4cd1d899af3f4747c7` | `pixel_values [N,3,256,256] float32` → `image_embeds [N,768]`, L2-normalised |
| `text.onnx` | 564,810,160 | `958aab2836acfcb43b431c5b8215a6b5b778b7998f81cdbd6ea5e0c7799355b0` | `input_ids [N,64] int64` → `text_embeds [N,768]`, L2-normalised |
| `tokenizer.model` | 4,241,003 | `61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2` | the Gemma SentencePiece tokenizer, unchanged from the checkpoint |
| `manifest.json` | — | — | the list above as JSON, with the download URLs |

## Using them

| | |
|---|---|
| Image input | resize to 256×256 without cropping, RGB, `(x / 255 − 0.5) / 0.5`, NCHW |
| Text input | Gemma SentencePiece ids; no BOS; EOS (id 1) appended; right-padded with 0 to 64; content truncated so the EOS fits; case kept as typed |
| Output | 768 floats, already unit length: the dot product is the cosine |
| Weights | float16 inside the graph; inputs and outputs float32; opset 17 |
| Fidelity | cosine ≥ 0.99999 against the PyTorch model on ten images and ten sentences |

SigLIP is trained with a sigmoid loss, so raw cosines are small (about 0.05 to
0.2 for a good match); rank by them rather than reading them as probabilities.

## Licence and changes

The model and tokenizer are © Google, released under the Apache License 2.0
(see [`LICENSE`](LICENSE) and the
[model card](https://huggingface.co/google/siglip2-base-patch16-256)).

Changes made here, as section 4(b) of the licence asks: the checkpoint was
converted to ONNX; the vision and text towers were split into separate graphs;
weights were cast to float16; L2 normalisation was added to the end of each
graph. The weights themselves were not retrained or altered beyond the cast.

This repository is not affiliated with or endorsed by Google.
