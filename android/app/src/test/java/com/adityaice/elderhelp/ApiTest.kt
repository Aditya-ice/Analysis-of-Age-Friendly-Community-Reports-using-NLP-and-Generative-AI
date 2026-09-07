package com.adityaice.elderhelp

import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.flow.toList
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import org.junit.Assert.*
import org.junit.Test
import java.util.concurrent.TimeUnit

class ApiTest {
    @Test fun multilineAndComments() {
        val parser = SseParser()
        assertNull(parser.line(": heartbeat")); parser.line("event: delta")
        parser.line("data: first"); parser.line("data: second")
        assertEquals(StreamEvent("delta", "first\nsecond"), parser.line(""))
        assertNull(parser.finish())
    }
    @Test fun fragmentedStream() = runTest {
        val server = MockWebServer(); server.start()
        try {
            server.enqueue(MockResponse().setHeader("Content-Type", "text/event-stream")
                .setBody("event: start\ndata: {}\n\nevent: delta\ndata: {\"text\":\"hello\"}\n\nevent: complete\ndata: {}\n\n")
                .throttleBody(3, 1, TimeUnit.MILLISECONDS))
            val events = Api(server.url("/").toString()).answer(AnswerRequest("Question")).toList()
            assertEquals(listOf("start", "delta", "complete"), events.map { it.name })
            assertEquals("POST", server.takeRequest().method)
        } finally { server.shutdown() }
    }
    @Test fun rejectsIncompleteStream() = runTest {
        val server = MockWebServer(); server.start()
        try {
            server.enqueue(MockResponse().setBody("event: delta\ndata: {}\n\n"))
            var failed = false
            try { Api(server.url("/").toString()).answer(AnswerRequest("Question")).toList() }
            catch (e: java.io.IOException) { failed = true }
            assertTrue(failed)
        } finally { server.shutdown() }
    }
    @Test fun decodesCitationContract() {
        val result = codec.decodeFromString<AnswerComplete>("""{"request_id":"request","answer_markdown":"Answer [S1]","status":"grounded","citations":[{"id":"S1","report_id":"report","report_title":"Title","publisher":"Publisher","source_url":"https://example.org","publication_date":null,"page_number":2,"excerpt":"Evidence"}]}""")
        assertEquals(2, result.citations.single().page_number)
    }
}
