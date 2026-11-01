# gemma-android-app

On-device Gemma 4 E2B Android client: **ONNX Runtime + Qualcomm QNN HTP** (Snapdragon 8 Gen 3).

## Models (this branch)

In-app download **prefers** the official **QAT mobile** ONNX suite:

- HF: [onnx-community/gemma-4-E2B-it-qat-mobile-ONNX](https://huggingface.co/onnx-community/gemma-4-E2B-it-qat-mobile-ONNX)
- Files: `embed_tokens_q2f16.*`, `decoder_model_merged_q2f16.*`, `audio_encoder_q2f16.*`

After download, the app also **aliases** those files to the legacy `*_q4f16.*` names so the existing `GemmaEngine` loader finds them without a large refactor.

If QAT download fails, it falls back to the previous ModelScope **q4f16** HTP package.

## HTP options (unchanged)

`enable_htp_fp16_precision`, `enable_block_quant_weight_optimization`, `htp_performance_mode=burst`, `htp_graph_finalization_optimization_mode=3`, `vtcm_mb=8`, arch 75 / soc 57.

## Build / artifact

```bash
./gradlew assembleDebug
```

GitHub Actions on push uploads `gemma-android-debug-apk-*`.

Branch: `feature/qat-mobile-clean`
