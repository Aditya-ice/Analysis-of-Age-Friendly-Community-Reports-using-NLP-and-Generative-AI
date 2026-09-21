package com.adityaice.elderhelp

import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.flow.take
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import org.junit.Assert.*
import org.junit.Test
import java.util.concurrent.TimeUnit

class ApiTest {
    private fun fixture(name: String) = requireNotNull(javaClass.classLoader?.getResourceAsStream("contracts/$name-v2.json")).bufferedReader().use { it.readText() }
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
                .setBody("event: start\ndata: {}\n\nevent: progress\ndata: ${fixture("progress").replace("\n", "")}\n\nevent: complete\ndata: ${fixture("complete").replace("\n", "")}\n\n")
                .throttleBody(3, 1, TimeUnit.MILLISECONDS))
            val events = Api(server.url("/").toString()).answer(AnswerRequest("Question")).toList()
            assertEquals(listOf("start", "progress", "complete"), events.map { it.name })
            val request = server.takeRequest()
            assertEquals("POST", request.method)
            assertEquals("/v2/answers/stream", request.path)
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
    @Test fun cancellationStopsCollection() = runTest {
        val server = MockWebServer(); server.start()
        try {
            server.enqueue(MockResponse().setBody("event: start\ndata: {}\n\n" + "event: delta\ndata: {}\n\n".repeat(200))
                .throttleBody(25, 20, TimeUnit.MILLISECONDS))
            val events = Api(server.url("/").toString()).answer(AnswerRequest("Question")).take(1).toList()
            assertEquals("start", events.single().name)
        } finally { server.shutdown() }
    }
    @Test fun rejectsUnavailableProvider() = runTest {
        val server = MockWebServer(); server.start()
        try {
            server.enqueue(MockResponse().setResponseCode(503))
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
    @Test fun pilotAccessAndSharedPartialContract() = runTest {
        val server = MockWebServer(); server.start()
        try {
            server.enqueue(MockResponse().setBody("""{"token":"test-token","expires_at":2000000000}"""))
            server.enqueue(MockResponse().setBody(fixture("capabilities")))
            val store = MemoryTokens()
            val api = Api(server.url("/").toString(), tokens = store)
            api.connect("mock-invite-code-only")
            assertTrue(api.capabilities().generation_available)
            assertEquals("/v2/demo/session", server.takeRequest().path)
            assertEquals("Bearer test-token", server.takeRequest().getHeader("Authorization"))
            assertEquals("test-token", store.read())
            api.forget(); assertNull(store.read())
            val result = codec.decodeFromString<AnswerComplete>(fixture("complete"))
            assertEquals("partial", result.status)
            assertEquals("2017", result.citations.single().publication_date)
            assertNotNull(result.citations.single().span_id)
        } finally { server.shutdown() }
    }

}
