package com.example.gemmaapp.engine

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class JavaScriptToolRuntimeTest {
    private val runtime = JavaScriptToolRuntime()

    @Test
    fun executesJavaScriptAndReportsPassingTests() {
        val result = runtime.execute(
            """
            function sum(a, b) { return a + b; }
            test('sum works', () => assert(sum(20, 22) === 42));
            console.log('red-black-tree harness ready');
            """.trimIndent()
        )
        assertTrue(result.success)
        assertTrue(result.output.contains("PASS sum works"))
        assertTrue(result.output.contains("red-black-tree harness ready"))
    }

    @Test
    fun reportsFailingTests() {
        val result = runtime.execute("test('bad invariant', () => assert(false, 'black height mismatch'));")
        assertFalse(result.success)
        assertTrue(result.output.contains("FAIL bad invariant"))
        assertTrue(result.output.contains("black height mismatch"))
    }

    @Test
    fun parsesNestedToolCallJson() {
        val parsed = ToolCallParser.parse(
            "<tool_call>{\"name\":\"javascript\",\"arguments\":{\"code\":\"test('x', () => assert(true));\"}}</tool_call>"
        )
        assertTrue(parsed?.code?.contains("test('x'") == true)
    }

    @Test
    fun parsesEscapedNewlinesAndQuotes() {
        val parsed = ToolCallParser.parse(
            """<tool_call>{"name":"javascript","arguments":{"code":"console.log(\"hello\");\ntest('line', () => assert(true));"}}</tool_call>"""
        )
        assertTrue(parsed?.code?.contains("console.log(\"hello\");") == true)
        assertTrue(parsed?.code?.contains("\ntest('line'") == true)
    }

    @Test
    fun removesToolCallFromVisibleText() {
        val visible = ToolCallParser.visibleText(
            "thinking<tool_call>{\"name\":\"javascript\",\"arguments\":{\"code\":\"1+1\"}}</tool_call>done"
        )
        assertTrue(visible == "thinkingdone")
    }
}
