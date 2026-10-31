package com.example.gemmaapp.ui

import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

@Composable
fun MarkdownMessage(content: String, textColor: Color) {
    val fenceMark = 96.toChar().toString().repeat(3)
    val fence = Regex("(?s)" + fenceMark + "([\\w+#.-]*)[ \\t]*\\r?\\n(.*?)" + fenceMark)
    val matches = fence.findAll(content).toList()
    var cursor = 0
    Column(verticalArrangement = Arrangement.spacedBy(5.dp)) {
        matches.forEach { match ->
            if (match.range.first > cursor) MarkdownTextBlock(content.substring(cursor, match.range.first), textColor)
            val language = match.groupValues[1].ifBlank { "code" }
            Surface(modifier = Modifier.fillMaxWidth(), color = MaterialTheme.colorScheme.surfaceVariant, shape = RoundedCornerShape(9.dp)) {
                Column {
                    Text(language.uppercase(), modifier = Modifier.padding(start = 10.dp, top = 7.dp), color = MaterialTheme.colorScheme.primary, fontSize = 10.sp, fontWeight = FontWeight.Bold)
                    SelectionContainer {
                        Text(match.groupValues[2].trimEnd(), modifier = Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()).padding(10.dp), color = MaterialTheme.colorScheme.onSurfaceVariant, fontFamily = FontFamily.Monospace, fontSize = 12.sp, lineHeight = 17.sp)
                    }
                }
            }
            cursor = match.range.last + 1
        }
        if (cursor < content.length) MarkdownTextBlock(content.substring(cursor), textColor)
    }
}

@Composable
private fun MarkdownTextBlock(block: String, color: Color) {
    block.lines().forEach { raw ->
        val line = raw.trimEnd()
        if (line.isBlank()) return@forEach
        val heading = Regex("^(#{1,3})\\s+(.+)$").matchEntire(line)
        val bullet = Regex("^\\s*[-*+]\\s+(.+)$").matchEntire(line)
        val ordered = Regex("^\\s*\\d+[.)]\\s+(.+)$").matchEntire(line)
        when {
            heading != null -> {
                val size = when (heading.groupValues[1].length) { 1 -> 19.sp; 2 -> 17.sp; else -> 15.sp }
                Text(inlineMarkdown(heading.groupValues[2], color), color = color, fontSize = size, fontWeight = FontWeight.Bold)
            }
            bullet != null -> Row(verticalAlignment = Alignment.Top) {
                Text("•  ", color = color)
                Text(inlineMarkdown(bullet.groupValues[1], color), color = color, fontSize = 14.sp, lineHeight = 20.sp)
            }
            ordered != null -> Text(inlineMarkdown(line.trimStart(), color), color = color, fontSize = 14.sp, lineHeight = 20.sp)
            line.trim() == "---" -> HorizontalDivider()
            else -> Text(inlineMarkdown(line, color), color = color, fontSize = 14.sp, lineHeight = 20.sp)
        }
    }
}

private fun inlineMarkdown(source: String, color: Color) = buildAnnotatedString {
    val tick = 96.toChar()
    val regex = Regex("(\\*\\*[^*]+\\*\\*|(?<!\\*)\\*[^*]+\\*(?!\\*)|~~[^~]+~~|" + tick + "[^" + tick + "]+" + tick + ")")
    var index = 0
    regex.findAll(source).forEach { match ->
        append(source.substring(index, match.range.first))
        val token = match.value
        when {
            token.startsWith("**") -> withStyle(SpanStyle(fontWeight = FontWeight.Bold)) { append(token.substring(2, token.length - 2)) }
            token.startsWith("~~") -> withStyle(SpanStyle(textDecoration = TextDecoration.LineThrough)) { append(token.substring(2, token.length - 2)) }
            token.first() == tick -> withStyle(SpanStyle(fontFamily = FontFamily.Monospace, background = color.copy(alpha = 0.12f))) { append(token.substring(1, token.length - 1)) }
            else -> withStyle(SpanStyle(fontStyle = FontStyle.Italic)) { append(token.substring(1, token.length - 1)) }
        }
        index = match.range.last + 1
    }
    append(source.substring(index))
}
