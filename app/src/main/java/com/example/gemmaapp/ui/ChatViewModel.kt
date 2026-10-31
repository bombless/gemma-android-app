package com.example.gemmaapp.ui

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.gemmaapp.engine.GemmaEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

class ChatViewModel(application: Application) : AndroidViewModel(application) {

    private val _messages = MutableStateFlow<List<ChatMessage>>(
        listOf(
            ChatMessage(
                sender = ChatMessage.Sender.SYSTEM,
                text = "欢迎使用 Gemma 4 掌上推理！基于 ONNX Runtime & Snapdragon 8 Gen 3 硬件加速。\n请输入问题开始对话。"
            )
        )
    )
    val messages: StateFlow<List<ChatMessage>> = _messages.asStateFlow()

    private val _isGenerating = MutableStateFlow(false)
    val isGenerating: StateFlow<Boolean> = _isGenerating.asStateFlow()

    private val _generationSpeed = MutableStateFlow(0f)
    val generationSpeed: StateFlow<Float> = _generationSpeed.asStateFlow()

    private val _generatedTokens = MutableStateFlow(0)
    val generatedTokens: StateFlow<Int> = _generatedTokens.asStateFlow()

    private val _backendStatus = MutableStateFlow("正在加载引擎与模型...")
    val backendStatus: StateFlow<String> = _backendStatus.asStateFlow()

    private var engine: GemmaEngine? = null

    init {
        loadEngine()
    }

    private fun loadEngine() {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                _backendStatus.value = "检测模型文件中..."
                val found = GemmaEngine.findModelFiles(getApplication())
                if (found == null) {
                    _backendStatus.value = "未找到模型文件，请下载至 /sdcard/Download/gemma-4-E2B-it-ONNX/"
                    return@launch
                }
                _backendStatus.value = "正在初始化 QNN HTP / CPU 引擎..."
                val eng = GemmaEngine.initialize(getApplication(), useQnnHtp = true)
                engine = eng
                _backendStatus.value = "运行后端: ${eng.backendName}"
            } catch (e: Exception) {
                _backendStatus.value = "引擎初始化失败: ${e.message}"
            }
        }
    }

    fun sendMessage(promptText: String) {
        if (promptText.isBlank() || _isGenerating.value) return

        val userMsg = ChatMessage(sender = ChatMessage.Sender.USER, text = promptText.trim())
        val initialModelMsg = ChatMessage(sender = ChatMessage.Sender.MODEL, text = "")

        _messages.value = _messages.value + userMsg + initialModelMsg
        _isGenerating.value = true
        _generationSpeed.value = 0f
        _generatedTokens.value = 0

        val currentEngine = engine
        if (currentEngine == null) {
            updateLastMessage("错误: Gemma 引擎尚未初始化成功，请检查模型文件并重试。")
            _isGenerating.value = false
            return
        }

        viewModelScope.launch(Dispatchers.IO) {
            val responseBuilder = StringBuilder()
            val generationStartNanos = System.nanoTime()
            try {
                val formatted = currentEngine.tokenizer.formatChatPrompt(promptText)
                currentEngine.generateStream(formatted).collect { chunk ->
                    responseBuilder.append(chunk.text)
                    _generationSpeed.value = chunk.tokensPerSecond
                    _generatedTokens.value = chunk.generatedTokens
                    updateLastMessage(responseBuilder.toString())
                }
            } catch (e: Exception) {
                responseBuilder.append("\n[生成异常]: ${e.message}")
            } finally {
                // Per-reply wall time includes prompt prefill and token decoding.
                val durationMs = ((System.nanoTime() - generationStartNanos) / 1_000_000L).coerceAtLeast(1L)
                val tokenCount = _generatedTokens.value
                val averageSpeed = tokenCount * 1000f / durationMs
                finishLastMessage(responseBuilder.toString(), durationMs, tokenCount, averageSpeed)
                _isGenerating.value = false
            }
        }
    }

    private fun updateLastMessage(newText: String) {
        val currentList = _messages.value.toMutableList()
        if (currentList.isNotEmpty()) {
            val last = currentList.last()
            if (last.sender == ChatMessage.Sender.MODEL) {
                currentList[currentList.size - 1] = last.copy(text = newText)
                _messages.value = currentList
            }
        }
    }

    private fun finishLastMessage(
        finalText: String,
        durationMs: Long,
        tokenCount: Int,
        averageSpeed: Float
    ) {
        val currentList = _messages.value.toMutableList()
        val lastIndex = currentList.lastIndex
        if (lastIndex >= 0 && currentList[lastIndex].sender == ChatMessage.Sender.MODEL) {
            currentList[lastIndex] = currentList[lastIndex].copy(
                text = finalText.ifEmpty { "（本次没有生成文本）" },
                generationDurationMs = durationMs,
                generatedTokens = tokenCount,
                averageTokensPerSecond = averageSpeed
            )
            _messages.value = currentList
        }
    }

    fun clearHistory() {
        _messages.value = listOf(
            ChatMessage(
                sender = ChatMessage.Sender.SYSTEM,
                text = "对话历史已清空。"
            )
        )
    }

    override fun onCleared() {
        super.onCleared()
        engine?.close()
    }
}
