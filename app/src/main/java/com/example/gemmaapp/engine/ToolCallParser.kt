package com.example.gemmaapp.engine

data class JavaScriptToolCall(val code: String)

/** Small, platform-independent parser for the one structured tool call we expose. */
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
        val keyToken = "\"" + key + "\""
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
            if (ch != '\\') {
                out.append(ch)
                continue
            }
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
                    out.append(codeUnit.toChar())
                    i += 4
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

    fun visibleText(text: String): String {
        val start = text.indexOf(OPEN_TAG)
        if (start < 0) return text.trim()
        val end = text.indexOf(CLOSE_TAG, start + OPEN_TAG.length)
        if (end < 0) return text.substring(0, start).trim()
        return (text.substring(0, start) + text.substring(end + CLOSE_TAG.length)).trim()
    }
}
