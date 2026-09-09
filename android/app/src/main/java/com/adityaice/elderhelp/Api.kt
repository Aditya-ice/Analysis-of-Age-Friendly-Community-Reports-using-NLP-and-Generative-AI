package com.adityaice.elderhelp

import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.channels.trySendBlocking
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.serialization.json.*
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.IOException
import java.util.concurrent.TimeUnit
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

val codec = Json { ignoreUnknownKeys = true; encodeDefaults = true }
data class StreamEvent(val name: String, val data: String)
interface TokenStore { fun read(): String?; fun save(token: String?) }
class MemoryTokens : TokenStore { private var token: String? = null; override fun read() = token; override fun save(token: String?) { this.token = token } }
class PilotFailure(val status: Int, retry: String = "60") : IOException(when(status) {
    401 -> "Pilot access expired or is invalid. Enter your invite code again. Saved history is still available."
    429 -> "The pilot reached a usage limit. Use keyword search, or retry after $retry seconds."
    413,422 -> "Check your question: use 1–2,000 characters and valid filters."
    else -> "The service is unavailable or waking from sleep. Wait a moment and retry."
})
fun connectionMessage(error: Throwable) = if (error is PilotFailure) error.message!! else "The connection failed or you are offline. Saved reports and history remain available. Retry when connected."

class SseParser {
    private var name = "message"
    private val data = mutableListOf<String>()
    private var length = 0
    fun line(line: String): StreamEvent? {
        if (line.isEmpty()) return finish()
        if (line.startsWith(":")) return null
        val field = line.substringBefore(':'); val value = line.substringAfter(':', "").removePrefix(" ")
        when (field) { "event" -> name = value; "data" -> { length += value.length; if(length > 131072) throw IOException("Oversized answer event"); data.add(value) } }
        return null
    }
    fun finish(): StreamEvent? {
        val event = if (data.isEmpty()) null else StreamEvent(name, data.joinToString("\n"))
        name = "message"; data.clear(); length = 0; return event
    }
}
class Api(private val base: String, private val client: OkHttpClient = OkHttpClient.Builder()
    .callTimeout(90, TimeUnit.SECONDS).readTimeout(65, TimeUnit.SECONDS).followRedirects(false).build(),
    private val tokens: TokenStore = MemoryTokens()) {
    @Volatile private var token: String? = tokens.read()
    fun hasAccess() = token != null
    fun forget() { token = null; tokens.save(null) }
    private fun request(path: String) = Request.Builder().url(base.trimEnd('/') + path).apply {
        token?.let { header("Authorization", "Bearer $it") }
    }
    private suspend fun load(request: Request): String = suspendCancellableCoroutine { continuation ->
        val call = client.newCall(request)
        continuation.invokeOnCancellation { call.cancel() }
        call.enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) { if(continuation.isActive) continuation.resumeWithException(e) }
            override fun onResponse(call: Call, response: Response) {
                try { response.use {
                    if (!it.isSuccessful) throw PilotFailure(it.code, it.header("Retry-After") ?: "60")
                    val body = it.body!!.string()
                    if(continuation.isActive) continuation.resume(body)
                } } catch(e: Exception) { if(continuation.isActive) continuation.resumeWithException(e) }
            }
        })
    }
    private fun post(path: String, json: String) = request(path).post(json.toRequestBody("application/json".toMediaType())).build()
    suspend fun connect(invite: String): PilotSession {
        val result = codec.decodeFromString<PilotSession>(load(post("/v2/demo/session", codec.encodeToString(SessionRequest.serializer(), SessionRequest(invite)))))
        tokens.save(result.token); token = result.token; return result
    }
    suspend fun capabilities() = codec.decodeFromString<Capabilities>(load(request("/v2/capabilities").build()))
    suspend fun reports(): ReportList {
        val all = mutableListOf<ReportSummary>()
        while(true) {
            val page = codec.decodeFromString<ReportList>(load(request("/v2/reports?offset=${all.size}&limit=100").build()))
            all.addAll(page.items)
            if(all.size >= page.total || page.items.isEmpty()) return ReportList(all, page.total, 0, 100)
        }
    }
    suspend fun report(id: String) = codec.decodeFromString<ReportDetail>(load(request("/v2/reports/$id").build()))
    suspend fun search(payload: AnswerRequest) = codec.decodeFromString<SearchResponse>(load(post("/v2/search", codec.encodeToString(AnswerRequest.serializer(), payload))))
    fun answer(payload: AnswerRequest) = callbackFlow {
        val call = client.newCall(post("/v2/answers/stream", codec.encodeToString(AnswerRequest.serializer(), payload)).newBuilder().header("Accept", "text/event-stream").build())
        call.enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) { close(e) }
            override fun onResponse(call: Call, response: Response) {
                try {
                    response.use {
                        if (!it.isSuccessful) throw PilotFailure(it.code, it.header("Retry-After") ?: "60")
                        val source = it.body!!.source(); val parser = SseParser(); var complete = false; var started = false; var delta = ""
                        while (!source.exhausted() && !call.isCanceled()) {
                            val event = parser.line(source.readUtf8LineStrict(131072)) ?: continue
                            if (event.name == "error") {
                                val code = codec.parseToJsonElement(event.data).jsonObject["code"]?.jsonPrimitive?.content
                                throw PilotFailure(if(code == "quota_exhausted") 429 else 503)
                            }
                            if(event.name == "start") { if(started) throw IOException("Duplicate start"); started = true }
                            else if(!started) throw IOException("Missing answer start")
                            if(event.name == "delta") { delta += codec.parseToJsonElement(event.data).jsonObject.getValue("text").jsonPrimitive.content; if(delta.length > 64000) throw IOException("Oversized answer") }
                            if(event.name == "complete") {
                                val result = codec.decodeFromString<AnswerComplete>(event.data)
                                if(delta.isNotEmpty() && delta != result.answer_markdown) throw IOException("Mismatched answer stream")
                                if(result.status !in listOf("grounded", "partial", "insufficient_evidence", "clarification_required")) throw IOException("Unknown answer status")
                                if(result.citations.any { c -> c.span_id == null || c.revision_id == null }) throw IOException("Missing source provenance")
                                complete = true
                            }
                            if(event.name !in listOf("start", "progress", "delta", "complete")) throw IOException("Unknown event")
                            if (trySendBlocking(event).isFailure) throw IOException("Answer stream interrupted")
                            if(complete) break
                        }
                        if (!complete && !call.isCanceled()) throw IOException("Connection ended before verification completed. Please retry.")
                        close()
                    }
                } catch (e: Exception) { close(e) }
            }
        })
        awaitClose { call.cancel() }
    }
}
