# gemma-android-app

On-device Gemma 4 E2B Android client using **ONNX Runtime + Qualcomm QNN HTP** (Snapdragon 8 Gen 3).

## Models

The app prefers the **official QAT mobile ONNX** suite:

- Source: [`onnx-community/gemma-4-E2B-it-qat-mobile-ONNX`](https://huggingface.co/onnx-community/gemma-4-E2B-it-qat-mobile-ONNX)
- Files (under `onnx/`):
  - `embed_tokens_q2f16.onnx` (+ `.onnx_data`)
  - `decoder_model_merged_q2f16.onnx` (+ `.onnx_data`)
  - `audio_encoder_q2f16.onnx` (+ `.onnx_data`)

This is Google’s QAT mobile format (mixed low-bit weights / static activations) aimed at phones.  
If download fails, the app falls back to the previous **q4f16** HTP package.

You can also place the same filenames under:

- `Android/data/<package>/files/models/`
- `/sdcard/Download/gemma-4-E2B-it-ONNX/`

## HTP notes

QNN options currently used (8 Gen 3 / HTP arch 75):

- `enable_htp_fp16_precision=1`
- `enable_block_quant_weight_optimization=1`
- `htp_performance_mode=burst`
- `htp_graph_finalization_optimization_mode=3`
- `vtcm_mb=8`

Related reading: Gemma 4 tech report, mobile NPU test-time scaling (arXiv:2509.23324), Quant.npu (arXiv:2605.20295).

## Build

```bash
./gradlew assembleDebug
```

GitHub Actions uploads a debug APK artifact on every push to `master` / this branch.

## Disclaimer

Research / prototype quality. QAT mobile graphs may differ slightly in tensor layout from the older q4f16 package; if HTP init fails, the app falls back to multi-threaded CPU.
