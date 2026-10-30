package com.example.gemmaapp.ui

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.gemmaapp.engine.GemmaEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.catch
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

        val currentEngine = engine
        if (currentEngine == null) {
            updateLastMessage("错误: Gemma 引擎尚未初始化成功，请检查模型文件并重试。")
            _isGenerating.value = false
            return
        }

        viewModelScope.launch(Dispatchers.IO) {
            val formatted = currentEngine.tokenizer.formatChatPrompt(promptText)
            val responseBuilder = StringBuilder()

            currentEngine.generateStream(formatted)
                .catch { e ->
                    responseBuilder.append("\n[生成异常]: ${e.message}")
                    updateLastMessage(responseBuilder.toString())
                    _isGenerating.value = false
                }
                .collect { token ->
                    responseBuilder.append(token)
                    updateLastMessage(responseBuilder.toString())
                }

            _isGenerating.value = false
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
