package com.adityaice.elderhelp

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.channels.trySendBlocking
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.*
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.IOException
import java.util.concurrent.TimeUnit

val codec = Json { ignoreUnknownKeys = true; encodeDefaults = true }
data class StreamEvent(val name: String, val data: String)

/** Line framing is independent of TCP fragmentation and supports multiline data. */
class SseParser {
    private var name = "message"
    private val data = mutableListOf<String>()
    fun line(line: String): StreamEvent? {
        if (line.isEmpty()) return finish()
        if (line.startsWith(":")) return null
        val field = line.substringBefore(':')
        val value = line.substringAfter(':', "").removePrefix(" ")
        when (field) { "event" -> name = value; "data" -> data.add(value) }
        return null
    }
    fun finish(): StreamEvent? {
        val event = if (data.isEmpty()) null else StreamEvent(name, data.joinToString("\n"))
        name = "message"; data.clear(); return event
    }
}
class Api(private val base: String, private val client: OkHttpClient = OkHttpClient.Builder()
    .callTimeout(60, TimeUnit.SECONDS).readTimeout(30, TimeUnit.SECONDS).build()) {
    private fun request(path: String) = Request.Builder().url(base.trimEnd('/') + path)
    suspend fun reports(): ReportList = withContext(Dispatchers.IO) {
        client.newCall(request("/v1/reports").build()).execute().use {
            if (!it.isSuccessful) throw IOException("Reports are unavailable (${it.code}). Try again.")
            codec.decodeFromString(it.body!!.string())
        }
    }
    fun answer(payload: AnswerRequest) = callbackFlow {
        val call = client.newCall(request("/v1/answers/stream")
            .header("Accept", "text/event-stream")
            .post(codec.encodeToString(AnswerRequest.serializer(), payload).toRequestBody("application/json".toMediaType())).build())
        call.enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) { close(e) }
            override fun onResponse(call: Call, response: Response) {
                try {
                    response.use {
                        if (!it.isSuccessful) throw IOException("Answer unavailable (${it.code}). Try again.")
                        val source = it.body!!.source(); val parser = SseParser(); var complete = false
                        while (!source.exhausted() && !call.isCanceled()) {
                            val event = parser.line(source.readUtf8Line() ?: break) ?: continue
                            if (event.name == "error") throw IOException("The answer service could not finish. Please retry.")
                            if (event.name == "complete") complete = true
                            if (trySendBlocking(event).isFailure) throw IOException("Answer stream interrupted.")
                        }
                        parser.finish()?.let { event ->
                            if (event.name == "complete") complete = true
                            trySendBlocking(event)
                        }
                        if (!complete && !call.isCanceled()) throw IOException("The connection ended before the answer finished. Please retry.")
                        close()
                    }
                } catch (e: Exception) { close(e) }
            }
        })
        awaitClose { call.cancel() }
    }
}
