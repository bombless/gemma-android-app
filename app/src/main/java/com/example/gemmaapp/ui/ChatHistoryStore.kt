package com.example.gemmaapp.ui

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.util.UUID

data class ChatSession(
    val id: String,
    val title: String,
    val updatedAt: Long,
    val messages: List<ChatMessage>
)

class ChatHistoryStore(context: Context) {
    private val prefs = context.getSharedPreferences("gemma_chat_history", Context.MODE_PRIVATE)

    fun load(): List<ChatSession> = runCatching {
        val array = JSONArray(prefs.getString(KEY, "[]") ?: "[]")
        (0 until array.length()).map { index ->
            val item = array.getJSONObject(index)
            val messageArray = item.optJSONArray("messages") ?: JSONArray()
            val messages = (0 until messageArray.length()).map { i ->
                val m = messageArray.getJSONObject(i)
                val tools = m.optJSONArray("toolCalls") ?: JSONArray()
                ChatMessage(
                    id = m.optString("id", UUID.randomUUID().toString()),
                    sender = ChatMessage.Sender.valueOf(m.optString("sender", "SYSTEM")),
                    text = m.optString("text"),
                    thinking = m.optString("thinking"),
                    toolCalls = (0 until tools.length()).map { j ->
                        val t = tools.getJSONObject(j)
                        ToolCallInfo(t.optString("name"), t.optString("arguments"), t.optString("code"),
                            t.optString("status", "成功"), t.optString("output"))
                    },
                    timestamp = m.optLong("timestamp", System.currentTimeMillis()),
                    generationDurationMs = if (m.isNull("duration")) null else m.optLong("duration"),
                    generatedTokens = if (m.isNull("tokens")) null else m.optInt("tokens"),
                    averageTokensPerSecond = if (m.isNull("speed")) null else m.optDouble("speed").toFloat()
                )
            }
            ChatSession(item.getString("id"), item.optString("title", "新对话"),
                item.optLong("updatedAt"), messages)
        }.sortedByDescending { it.updatedAt }
    }.getOrDefault(emptyList())

    fun save(sessions: List<ChatSession>) {
        val array = JSONArray()
        sessions.forEach { session ->
            val item = JSONObject().put("id", session.id).put("title", session.title)
                .put("updatedAt", session.updatedAt)
            val messages = JSONArray()
            session.messages.forEach { message ->
                val tools = JSONArray()
                message.toolCalls.forEach { tool ->
                    tools.put(JSONObject().put("name", tool.name).put("arguments", tool.arguments)
                        .put("code", tool.code).put("status", tool.status).put("output", tool.output))
                }
                messages.put(JSONObject().put("id", message.id).put("sender", message.sender.name)
                    .put("text", message.text).put("thinking", message.thinking).put("toolCalls", tools)
                    .put("timestamp", message.timestamp).put("duration", message.generationDurationMs)
                    .put("tokens", message.generatedTokens).put("speed", message.averageTokensPerSecond))
            }
            item.put("messages", messages)
            array.put(item)
        }
        prefs.edit().putString(KEY, array.toString()).apply()
    }

    companion object { private const val KEY = "sessions_v1" }
}
