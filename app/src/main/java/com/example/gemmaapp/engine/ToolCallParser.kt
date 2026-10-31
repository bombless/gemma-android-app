package com.example.gemmaapp.engine

data class JavaScriptToolCall(val code: String)

/** Parser for the structured JavaScript tool call and Gemma reasoning channels. */
object ToolCallParser {
    private const val OPEN_TAG = "<tool_call>"
    private const val CLOSE_TAG = "</tool_call>"

    fun parse(text: String): JavaScriptToolCall? {
        val start = text.indexOf(OPEN_TAG)
        if (start < 0) return null
        val jsonStart = start + OPEN_TAG.length
        val end = text.indexOf(CLOSE_TAG, jsonStart)
        if (end < 0) return null
        val json = text.substring(jsonStart, end)
        val name = readJsonString(json, "name") ?: return null
        if (name != "javascript") return null
        val code = readJsonString(json, "code") ?: return null
        return if (code.isBlank()) null else JavaScriptToolCall(code)
    }

    private fun readJsonString(json: String, key: String): String? {
        val keyToken = "\"$key\""
        val keyStart = json.indexOf(keyToken)
        if (keyStart < 0) return null
        var i = keyStart + keyToken.length
        while (i < json.length && json[i].isWhitespace()) i++
        if (i >= json.length || json[i++] != ':') return null
        while (i < json.length && json[i].isWhitespace()) i++
        if (i >= json.length || json[i++] != '"') return null
        val encoded = StringBuilder()
        var escaped = false
        while (i < json.length) {
            val ch = json[i++]
            if (escaped) {
                encoded.append('\\').append(ch)
                escaped = false
            } else when (ch) {
                '\\' -> escaped = true
                '"' -> return decodeJsonString(encoded.toString())
                else -> encoded.append(ch)
            }
        }
        return null
    }

    private fun decodeJsonString(value: String): String? {
        val out = StringBuilder()
        var i = 0
        while (i < value.length) {
            val ch = value[i++]
            if (ch != '\\') { out.append(ch); continue }
            if (i >= value.length) return null
            when (val escaped = value[i++]) {
                '"', '\\', '/' -> out.append(escaped)
                'b' -> out.append('\b')
                'f' -> out.append('\u000C')
                'n' -> out.append('\n')
                'r' -> out.append('\r')
                't' -> out.append('\t')
                'u' -> {
                    if (i + 4 > value.length) return null
                    val codeUnit = value.substring(i, i + 4).toIntOrNull(16) ?: return null
                    out.append(codeUnit.toChar()); i += 4
                }
                else -> return null
            }
        }
        return out.toString()
    }

    fun validateParser() {
        check(parse("<tool_call>{\"name\":\"javascript\",\"arguments\":{\"code\":\"console.log('ok');\"}}</tool_call>")?.code == "console.log('ok');") {
            "Tool-call parser self-test failed"
        }
    }

    fun thinkingText(text: String): String {
        val thinkMarker = "<|think|>"
        val thinkStart = text.indexOf(thinkMarker)
        if (thinkStart >= 0) {
            val from = thinkStart + thinkMarker.length
            val ends = listOf("<|channel>final", "<|channel>analysis", OPEN_TAG)
                .map { text.indexOf(it, from) }.filter { it >= 0 }
            return text.substring(from, ends.minOrNull() ?: text.length).trim()
        }
        val channel = "<|channel>analysis"
        val channelStart = text.indexOf(channel)
        if (channelStart >= 0) {
            val headerEnd = text.indexOf("<channel|>", channelStart)
            val newline = text.indexOf('\n', channelStart)
            val from = when {
                headerEnd >= 0 && (newline < 0 || headerEnd < newline) -> headerEnd + "<channel|>".length
                newline >= 0 -> newline + 1
                else -> channelStart + channel.length
            }
            val ends = listOf("<|channel>final", OPEN_TAG)
                .map { text.indexOf(it, from) }.filter { it >= 0 }
            return text.substring(from, ends.minOrNull() ?: text.length).trim()
        }
        return ""
    }

    fun toolCallPreview(text: String): String? {
        val start = text.indexOf(OPEN_TAG)
        if (start < 0) return null
        val from = start + OPEN_TAG.length
        val end = text.indexOf(CLOSE_TAG, from)
        return text.substring(from, if (end >= 0) end else text.length).trim().takeIf { it.isNotEmpty() }
    }

    fun visibleText(text: String): String {
        var result = text
        val thinkStart = result.indexOf("<|think|>")
        if (thinkStart >= 0) {
            val ends = listOf("<|channel>final", "<|channel>analysis")
                .map { result.indexOf(it, thinkStart) }.filter { it >= 0 }
            val end = ends.minOrNull()
            result = if (end != null) result.removeRange(thinkStart, end) else result.substring(0, thinkStart)
        }
        val analysisStart = result.indexOf("<|channel>analysis")
        if (analysisStart >= 0) {
            val finalStart = result.indexOf("<|channel>final", analysisStart)
            val toolStart = result.indexOf(OPEN_TAG, analysisStart)
            val end = listOf(finalStart, toolStart).filter { it >= 0 }.minOrNull()
            result = if (end != null) result.removeRange(analysisStart, end) else result.substring(0, analysisStart)
        }
        val start = result.indexOf(OPEN_TAG)
        if (start >= 0) {
            val end = result.indexOf(CLOSE_TAG, start + OPEN_TAG.length)
            result = if (end < 0) result.substring(0, start) else result.removeRange(start, end + CLOSE_TAG.length)
        }
        return result.replace("<|channel>final", "").replace("<channel|>", "").trim()
    }
}
