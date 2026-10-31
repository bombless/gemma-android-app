package com.example.gemmaapp.ui

data class ChatMessage(
    val id: String = java.util.UUID.randomUUID().toString(),
    val sender: Sender,
    val text: String,
    val timestamp: Long = System.currentTimeMillis(),
    val generationDurationMs: Long? = null,
    val generatedTokens: Int? = null,
    val averageTokensPerSecond: Float? = null
) {
    enum class Sender {
        USER,
        MODEL,
        SYSTEM
    }
}
