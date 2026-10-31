package com.example.gemmaapp.ui

import android.app.Application
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.gemmaapp.engine.GemmaEngine
import com.example.gemmaapp.engine.JavaScriptToolRuntime
import com.example.gemmaapp.engine.ToolCallParser
import com.example.gemmaapp.engine.ModelDownloadProgress
import com.example.gemmaapp.engine.ModelScopeModelDownloader
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

    private val _downloadProgress = MutableStateFlow<ModelDownloadProgress?>(null)
    val downloadProgress: StateFlow<ModelDownloadProgress?> = _downloadProgress.asStateFlow()

    private val _isDownloadingModels = MutableStateFlow(false)
    val isDownloadingModels: StateFlow<Boolean> = _isDownloadingModels.asStateFlow()

    private val _modelDownloadFailed = MutableStateFlow(false)
    val modelDownloadFailed: StateFlow<Boolean> = _modelDownloadFailed.asStateFlow()

    private val _isEngineReady = MutableStateFlow(false)
    val isEngineReady: StateFlow<Boolean> = _isEngineReady.asStateFlow()

    private var engine: GemmaEngine? = null
    private val javaScriptRuntime = JavaScriptToolRuntime()
    private var isLoadingEngine = false

    private val toolSystemPrompt = """
        你是运行在 Android 手机上的 Gemma 编程助手。你可以通过 javascript 工具执行 JavaScript，并根据真实执行结果修复代码。
        当用户要求编写算法、调试代码或验证正确性时，优先编写可执行代码并调用工具测试，不要只凭推理声称测试通过。
        工具调用必须严格使用以下格式，JSON 必须合法，代码放在 arguments.code 字符串中（换行写成 \n，双引号和反斜杠按 JSON 规则转义）：
        <tool_call>{"name":"javascript","arguments":{"code":"console.log('hello'); test('basic', () => assert(1 + 1 === 2));"}}</tool_call>
        工具可用 API：
        - console.log(...): 输出日志
        - assert(condition, message): 条件不成立时抛出错误
        - test(name, () => { ... }): 执行一个测试并记录 PASS/FAIL；测试内使用 assert
        - tools.randomInt(min, max): 生成包含边界的随机整数
        - tools.now(): 当前 Unix 毫秒时间戳
        - tools.assert / tools.test: 上述函数的 tools 命名空间别名
        JavaScript 使用 ES6。运行环境是隔离沙箱，不提供文件、网络、Android、Java 类访问。
        可以在一次测试脚本中实现算法、构造随机数据、检查不变量并打印汇总。测试失败时应分析错误，再发起下一次工具调用。
        每次只发起一个 javascript 工具调用。拿到工具结果后再回答。最多进行 3 轮工具调用。
        如果无需工具，直接用中文回答。工具调用标签外不要输出解释文字。
    """.trimIndent()

    init {
        loadEngine()
    }

    private fun loadEngine() {
        if (isLoadingEngine || engine != null) return
        isLoadingEngine = true
        Log.i(TAG, "Engine initialization started")
        viewModelScope.launch(Dispatchers.IO) {
            try {
                _modelDownloadFailed.value = false
                _backendStatus.value = "检查本地模型文件..."
                var found = GemmaEngine.findModelFiles(getApplication())
                if (found == null) {
                    _isDownloadingModels.value = true
                    _backendStatus.value = "正在从 ModelScope 下载 Gemma 4 模型..."
                    ModelScopeModelDownloader.download(getApplication()) { progress ->
                        _downloadProgress.value = progress
                    }
                    found = GemmaEngine.findModelFiles(getApplication())
                    if (found == null) throw IllegalStateException("模型下载完成，但未找到完整模型文件")
                }
                _backendStatus.value = "正在初始化 QNN HTP / CPU 引擎..."
                val eng = GemmaEngine.initialize(getApplication(), useQnnHtp = true)
                engine = eng
                _isEngineReady.value = true
                _backendStatus.value = "运行后端: ${eng.backendName}"
                Log.i(TAG, "Engine ready backend=${eng.backendName}")
            } catch (e: Exception) {
                Log.e(TAG, "Engine initialization failed", e)
                _modelDownloadFailed.value = GemmaEngine.findModelFiles(getApplication()) == null
                _backendStatus.value = if (_modelDownloadFailed.value) {
                    "模型下载失败: ${e.message ?: "未知错误"}"
                } else {
                    "引擎初始化失败: ${e.message}"
                }
            } finally {
                _isDownloadingModels.value = false
                isLoadingEngine = false
            }
        }
    }

    fun retryModelDownload() {
        if (engine == null && !_isDownloadingModels.value) loadEngine()
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
            updateLastMessage("错误: Gemma 引擎尚未初始化成功，请等待模型下载完成或点击重试。")
            _isGenerating.value = false
            return
        }

        viewModelScope.launch(Dispatchers.IO) {
            val responseBuilder = StringBuilder()
            val generationStartNanos = System.nanoTime()
            var totalTokens = 0
            var conversationPrompt = currentEngine.tokenizer.formatChatPrompt(promptText, toolSystemPrompt)
            try {
                var toolRounds = 0
                while (true) {
                    val roundBuilder = StringBuilder()
                    currentEngine.generateStream(conversationPrompt, maxNewTokens = 768).collect { chunk ->
                        roundBuilder.append(chunk.text)
                        totalTokens += 1
                        _generationSpeed.value = chunk.tokensPerSecond
                        _generatedTokens.value = totalTokens
                        val visible = ToolCallParser.visibleText(roundBuilder.toString())
                        updateLastMessage((responseBuilder.toString() + visible).ifBlank { "Gemma 正在思考..." })
                    }

                    val roundText = roundBuilder.toString()
                    Log.d(TAG, "Generation round completed chars=${roundText.length}")
                    val toolCall = try {
                        ToolCallParser.parse(roundText)
                    } catch (t: Throwable) {
                        Log.e(TAG, "Tool-call parser failed", t)
                        throw t
                    }
                    if (toolCall == null || toolRounds >= 3) {
                        responseBuilder.append(ToolCallParser.visibleText(roundText))
                        break
                    }

                    toolRounds++
                    Log.i(TAG, "JavaScript tool call round=$toolRounds codeChars=${toolCall.code.length}")
                    val preamble = ToolCallParser.visibleText(roundText)
                    if (preamble.isNotBlank()) responseBuilder.append(preamble).append("\n\n")
                    updateLastMessage(responseBuilder.toString() + "🛠 正在执行 JavaScript 工具（第 $toolRounds 轮）...")
                    val toolResult = javaScriptRuntime.execute(toolCall.code)
                    Log.i(TAG, "JavaScript tool completed success=${toolResult.success} outputChars=${toolResult.output.length}")
                    val status = if (toolResult.success) "success" else "error"
                    responseBuilder.append("**JavaScript 工具执行（第 $toolRounds 轮，$status）**\n")
                        .append(toolResult.output).append("\n\n")
                    updateLastMessage(responseBuilder.toString())

                    // Feed the actual tool output back as a user/tool-result turn so Gemma can inspect it.
                    conversationPrompt += "<|turn>model\n$roundText<turn|>\n" +
                        "<|turn>user\n<tool_result name=\"javascript\" status=\"$status\">\n" +
                        toolResult.output.take(12_000) + "\n</tool_result><turn|>\n<|turn>model\n"
                }
            } catch (t: Throwable) {
                Log.e(TAG, "Generation/tool loop crashed", t)
                responseBuilder.append("\n[生成异常]: ${t.message ?: t.javaClass.simpleName}")
            } finally {
                val durationMs = ((System.nanoTime() - generationStartNanos) / 1_000_000L).coerceAtLeast(1L)
                val averageSpeed = totalTokens * 1000f / durationMs
                finishLastMessage(responseBuilder.toString(), durationMs, totalTokens, averageSpeed)
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

    companion object {
        private const val TAG = "GemmaChat"
    }
}
