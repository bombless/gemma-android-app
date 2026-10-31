package com.example.gemmaapp.engine

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtLoggingLevel
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.LongBuffer
import java.nio.ShortBuffer
import kotlin.math.exp

data class GenerationChunk(
    val text: String,
    val generatedTokens: Int,
    val tokensPerSecond: Float
)

/**
 * OnnxRuntime execution engine for Gemma 4 ONNX on Snapdragon 8 Gen 3.
 * Supports QNN HTP (NPU) hardware acceleration or high-speed multi-threaded CPU fallback.
 * KV cache and logits are natively in Float16 (ShortBuffer).
 */
class GemmaEngine private constructor(
    private val env: OrtEnvironment,
    private val embedSession: OrtSession,
    private val decoderSession: OrtSession,
    val tokenizer: GemmaTokenizer,
    val backendName: String
) {
    companion object {
        private const val TAG = "GemmaEngine"
        private const val NUM_KV_LAYERS = 15 // Gemma 4 E2B merged decoder KV layers (0..14)
        private const val MAX_SEQ_LEN = 2048

        // KV cache dimensions:
        // Layers with head_dim=512: 4, 9, 14
        // Other layers with head_dim=256: 0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13
        private val LARGE_HEAD_LAYERS = setOf(4, 9, 14)

        fun getHeadDim(layer: Int): Int = if (layer in LARGE_HEAD_LAYERS) 512 else 256

        data class ModelPaths(
            val embedModelFile: File,
            val embedDataFile: File,
            val decoderModelFile: File,
            val decoderDataFile: File
        )

        fun findModelFiles(context: Context): ModelPaths? {
            val candidateDirs = listOfNotNull(
                File(context.getExternalFilesDir(null), "models"),
                context.getExternalFilesDir(null),
                File(context.filesDir, "models"),
                context.filesDir,
                File("/sdcard/Download/gemma-4-E2B-it-ONNX"),
                File("/sdcard/Download"),
                File("/data/local/tmp/models")
            )

            for (dir in candidateDirs) {
                val embedModel = File(dir, "embed_tokens_q4f16.onnx")
                val embedData = File(dir, "embed_tokens_q4f16.onnx_data")
                val decoderModel = File(dir, "decoder_model_merged_q4f16.onnx")
                val decoderData = File(dir, "decoder_model_merged_q4f16.onnx_data")

                Log.i(TAG, "Scanning for models in ${dir.absolutePath}...")
                if (embedModel.exists() && embedData.exists() && decoderModel.exists() && decoderData.exists()) {
                    Log.i(TAG, "Found complete ONNX model suite in: ${dir.absolutePath}")
                    return ModelPaths(embedModel, embedData, decoderModel, decoderData)
                }
            }
            return null
        }

        fun initialize(context: Context, useQnnHtp: Boolean = true): GemmaEngine {
            val tokenizer = GemmaTokenizer.load(context)
            val modelPaths = findModelFiles(context)
                ?: throw IllegalStateException("Gemma ONNX model files not found. Please place embed and decoder ONNX files in /sdcard/Download/gemma-4-E2B-it-ONNX/")

            val env = OrtEnvironment.getEnvironment(OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING, "GemmaApp")

            var backend = "CPU"
            var decoderSession: OrtSession? = null
            var embedSession: OrtSession? = null

            if (useQnnHtp) {
                try {
                    Log.i(TAG, "Attempting to initialize QNN HTP Execution Provider for 8Gen3...")
                    QnnEpRegistration.ensureRegistered(env)
                    val device = env.epDevices.firstOrNull { it.epName == QnnEpRegistration.EP_NAME }
                    val backendLib = File(context.applicationInfo.nativeLibraryDir, "libQnnHtp.so")

                    if (device != null && backendLib.isFile) {
                        val qnnOpts = OrtSession.SessionOptions().apply {
                            setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING)
                            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                            addExecutionProvider(
                                listOf(device),
                                mapOf(
                                    "backend_path" to backendLib.absolutePath,
                                    "soc_model" to "57",
                                    "htp_arch" to "75",
                                    "enable_htp_fp16_precision" to "1",
                                    "enable_block_quant_weight_optimization" to "1",
                                    "htp_performance_mode" to "burst",
                                    "qnn_context_priority" to "high",
                                    "vtcm_mb" to "8",
                                    "htp_graph_finalization_optimization_mode" to "3"
                                )
                            )
                        }
                        decoderSession = env.createSession(modelPaths.decoderModelFile.absolutePath, qnnOpts)
                        embedSession = env.createSession(modelPaths.embedModelFile.absolutePath, qnnOpts)
                        backend = "QNN HTP (Snapdragon 8 Gen 3 NPU)"
                        Log.i(TAG, "Successfully initialized with QNN HTP hardware accelerator!")
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "QNN EP initialization failed, falling back to multi-threaded CPU: ${e.message}")
                }
            }

            if (decoderSession == null || embedSession == null) {
                Log.i(TAG, "Initializing CPU Execution Provider (8 cores)...")
                val cpuOpts = OrtSession.SessionOptions().apply {
                    setIntraOpNumThreads(8)
                    setInterOpNumThreads(2)
                    setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                }
                decoderSession = env.createSession(modelPaths.decoderModelFile.absolutePath, cpuOpts)
                embedSession = env.createSession(modelPaths.embedModelFile.absolutePath, cpuOpts)
                backend = "CPU (ARM64 FP16/NEON 8-Core)"
            }

            return GemmaEngine(env, embedSession, decoderSession, tokenizer, backend)
        }
    }

    private class LayerKvBuffer(val headDim: Int, val maxSeqLen: Int) {
        val keyBuf: ShortBuffer = ByteBuffer.allocateDirect(1 * 1 * maxSeqLen * headDim * 2)
            .order(ByteOrder.nativeOrder()).asShortBuffer()
        val valBuf: ShortBuffer = ByteBuffer.allocateDirect(1 * 1 * maxSeqLen * headDim * 2)
            .order(ByteOrder.nativeOrder()).asShortBuffer()

        fun reset() {
            keyBuf.clear()
            valBuf.clear()
        }
    }

    private val kvBuffers = Array(NUM_KV_LAYERS) { l ->
        LayerKvBuffer(getHeadDim(l), MAX_SEQ_LEN)
    }

    fun generateStream(
        prompt: String,
        maxNewTokens: Int = 512,
        temperature: Float = 0.7f,
        topP: Float = 0.9f
    ): Flow<GenerationChunk> = flow {
        val inputTokens = tokenizer.encode(prompt)
        if (inputTokens.isEmpty()) return@flow

        for (buf in kvBuffers) buf.reset()

        val promptLen = inputTokens.size
        var currentTotalSeqLen = promptLen

        // 1. Prefill stage: run embed_tokens on prompt input_ids
        val inputIdsLongArray = LongArray(promptLen) { i -> inputTokens[i].toLong() }
        val inputIdsTensor = OnnxTensor.createTensor(
            env,
            LongBuffer.wrap(inputIdsLongArray),
            longArrayOf(1, promptLen.toLong())
        )

        val embedInputs = mapOf("input_ids" to inputIdsTensor)
        val embedResults = embedSession.run(embedInputs)
        val inputsEmbeds = embedResults.get("inputs_embeds").get() as OnnxTensor
        val perLayerInputs = embedResults.get("per_layer_inputs").get() as OnnxTensor

        // Run decoder for prefill
        val posIdsArray = LongArray(promptLen) { i -> i.toLong() }
        val attnMaskArray = LongArray(promptLen) { 1L }

        val posIdsTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(posIdsArray), longArrayOf(1, promptLen.toLong()))
        val attnMaskTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(attnMaskArray), longArrayOf(1, promptLen.toLong()))
        val numLogitsTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(1L)), longArrayOf())

        val decoderInputs = HashMap<String, OnnxTensor>()
        decoderInputs["inputs_embeds"] = inputsEmbeds
        decoderInputs["attention_mask"] = attnMaskTensor
        decoderInputs["position_ids"] = posIdsTensor
        decoderInputs["num_logits_to_keep"] = numLogitsTensor
        decoderInputs["per_layer_inputs"] = perLayerInputs

        // Initial empty past_key_values (past_sequence_length = 0) with FLOAT16 type
        for (l in 0 until NUM_KV_LAYERS) {
            val hDim = getHeadDim(l).toLong()
            val emptyTensor = OnnxTensor.createTensor(
                env,
                ShortBuffer.wrap(ShortArray(0)),
                longArrayOf(1, 1, 0, hDim),
                OnnxJavaType.FLOAT16
            )
            decoderInputs["past_key_values.$l.key"] = emptyTensor
            decoderInputs["past_key_values.$l.value"] = emptyTensor
        }

        var decResult = decoderSession.run(decoderInputs)
        var lastLogitsTensor = decResult.get("logits").get() as OnnxTensor

        // Save present KV into our buffers
        for (l in 0 until NUM_KV_LAYERS) {
            val presKey = decResult.get("present.$l.key").get() as OnnxTensor
            val presVal = decResult.get("present.$l.value").get() as OnnxTensor
            val kSb = presKey.shortBuffer
            val vSb = presVal.shortBuffer
            val kb = kvBuffers[l].keyBuf
            val vb = kvBuffers[l].valBuf
            kb.clear(); kb.put(kSb); kb.flip()
            vb.clear(); vb.put(vSb); vb.flip()
        }

        // Measure decode throughput separately from prompt prefill latency.
        val decodeStartNanos = System.nanoTime()
        var generatedTokens = 0

        // Generate tokens sequentially
        for (step in 0 until maxNewTokens) {
            val nextToken = sampleNextTokenFp16(lastLogitsTensor, temperature, topP)
            if (nextToken == GemmaTokenizer.EOS_TOKEN_ID || nextToken == (tokenizer.vocab["<turn|>"] ?: 107)) {
                break
            }

            val tokenText = tokenizer.decodeToken(nextToken)
            if (tokenText.contains("<turn|>") || tokenText.contains("<eos>")) {
                break
            }
            generatedTokens += 1
            val elapsedSeconds = ((System.nanoTime() - decodeStartNanos).coerceAtLeast(1L)) / 1_000_000_000.0
            emit(
                GenerationChunk(
                    text = tokenText,
                    generatedTokens = generatedTokens,
                    tokensPerSecond = (generatedTokens / elapsedSeconds).toFloat()
                )
            )

            currentTotalSeqLen += 1
            if (currentTotalSeqLen >= MAX_SEQ_LEN) break

            // Embed single token
            val singleIdTensor = OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(longArrayOf(nextToken.toLong())),
                longArrayOf(1, 1)
            )

            val stepEmbedRes = embedSession.run(mapOf("input_ids" to singleIdTensor))
            val stepInputsEmbeds = stepEmbedRes.get("inputs_embeds").get() as OnnxTensor
            val stepPerLayerInputs = stepEmbedRes.get("per_layer_inputs").get() as OnnxTensor

            val stepPosTensor = OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(longArrayOf((currentTotalSeqLen - 1).toLong())),
                longArrayOf(1, 1)
            )

            val stepMaskArray = LongArray(currentTotalSeqLen) { 1L }
            val stepMaskTensor = OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(stepMaskArray),
                longArrayOf(1, currentTotalSeqLen.toLong())
            )

            val stepDecInputs = HashMap<String, OnnxTensor>()
            stepDecInputs["inputs_embeds"] = stepInputsEmbeds
            stepDecInputs["attention_mask"] = stepMaskTensor
            stepDecInputs["position_ids"] = stepPosTensor
            stepDecInputs["num_logits_to_keep"] = numLogitsTensor
            stepDecInputs["per_layer_inputs"] = stepPerLayerInputs

            val pastLen = (currentTotalSeqLen - 1).toLong()
            for (l in 0 until NUM_KV_LAYERS) {
                val hDim = getHeadDim(l).toLong()
                val kb = kvBuffers[l].keyBuf.duplicate()
                val vb = kvBuffers[l].valBuf.duplicate()
                kb.position(0); kb.limit((pastLen * hDim).toInt())
                vb.position(0); vb.limit((pastLen * hDim).toInt())

                stepDecInputs["past_key_values.$l.key"] = OnnxTensor.createTensor(
                    env,
                    kb,
                    longArrayOf(1, 1, pastLen, hDim),
                    OnnxJavaType.FLOAT16
                )
                stepDecInputs["past_key_values.$l.value"] = OnnxTensor.createTensor(
                    env,
                    vb,
                    longArrayOf(1, 1, pastLen, hDim),
                    OnnxJavaType.FLOAT16
                )
            }

            decResult = decoderSession.run(stepDecInputs)
            lastLogitsTensor = decResult.get("logits").get() as OnnxTensor

            for (l in 0 until NUM_KV_LAYERS) {
                val presKey = decResult.get("present.$l.key").get() as OnnxTensor
                val presVal = decResult.get("present.$l.value").get() as OnnxTensor
                val kSb = presKey.shortBuffer
                val vSb = presVal.shortBuffer
                val kb = kvBuffers[l].keyBuf
                val vb = kvBuffers[l].valBuf
                kb.clear(); kb.put(kSb); kb.flip()
                vb.clear(); vb.put(vSb); vb.flip()
            }
        }
    }.flowOn(Dispatchers.Default)

    /**
     * Converts an IEEE 754 half-precision float (16-bit short) to standard 32-bit float.
     */
    private fun fp16ToFloat(hbits: Short): Float {
        var mant = (hbits.toInt() and 0x03FF)
        var exp = (hbits.toInt() and 0x7C00)
        if (exp == 0x7C00) {
            exp = 0x3FC00
        } else if (exp != 0) {
            exp += 0x1C000
            if (mant == 0 && exp > 0x1C400) {
                return Float.fromBits(((hbits.toInt() and 0x8000) shl 16) or (exp shl 13))
            }
        } else if (mant != 0) {
            exp = 0x1C400
            do {
                mant = mant shl 1
                exp -= 0x400
            } while ((mant and 0x400) == 0)
            mant = mant and 0x3FF
        }
        return Float.fromBits(((hbits.toInt() and 0x8000) shl 16) or ((exp or mant) shl 13))
    }

    private fun sampleNextTokenFp16(logitsTensor: OnnxTensor, temperature: Float, topP: Float): Int {
        val sb = logitsTensor.shortBuffer
        val vocabSize = sb.remaining()
        val logits = FloatArray(vocabSize)
        val shortArr = ShortArray(vocabSize)
        sb.get(shortArr)

        for (i in 0 until vocabSize) {
            logits[i] = fp16ToFloat(shortArr[i])
        }

        if (temperature <= 0.001f) {
            var maxIdx = 0
            var maxVal = logits[0]
            for (i in 1 until vocabSize) {
                if (logits[i] > maxVal) {
                    maxVal = logits[i]
                    maxIdx = i
                }
            }
            return maxIdx
        }

        var maxLogit = Float.NEGATIVE_INFINITY
        for (i in 0 until vocabSize) {
            logits[i] /= temperature
            if (logits[i] > maxLogit) maxLogit = logits[i]
        }

        var sumExp = 0.0
        val probs = DoubleArray(vocabSize)
        for (i in 0 until vocabSize) {
            val p = exp((logits[i] - maxLogit).toDouble())
            probs[i] = p
            sumExp += p
        }

        val r = Math.random() * sumExp
        var accum = 0.0
        for (i in 0 until vocabSize) {
            accum += probs[i]
            if (accum >= r) return i
        }
        return vocabSize - 1
    }

    fun close() {
        embedSession.close()
        decoderSession.close()
        env.close()
    }
}
