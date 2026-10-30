package com.example.gemmaapp.ui

data class ChatMessage(
    val id: String = java.util.UUID.randomUUID().toString(),
    val sender: Sender,
    val text: String,
    val timestamp: Long = System.currentTimeMillis()
) {
    enum class Sender {
        USER,
        MODEL,
        SYSTEM
    }
}
