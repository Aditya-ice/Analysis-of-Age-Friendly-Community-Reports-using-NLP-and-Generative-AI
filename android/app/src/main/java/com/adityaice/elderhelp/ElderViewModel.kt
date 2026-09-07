package com.adityaice.elderhelp

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import kotlinx.serialization.json.*

data class AskState(val question: String = "", val text: String = "", val result: AnswerComplete? = null,
    val loading: Boolean = false, val error: String? = null, val reportId: String? = null)
class ElderViewModel(private val repo: Repository) : ViewModel() {
    val ask = MutableStateFlow(AskState())
    val libraryError = MutableStateFlow<String?>(null)
    val history = repo.store.history().stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())
    val reports = repo.store.reports().map { rows -> rows.map { codec.decodeFromString<ReportSummary>(it.payload) } }
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())
    private var job: Job? = null
    private val turns = mutableListOf<ChatTurn>()
    init { refresh() }
    fun question(value: String) { if (!ask.value.loading) ask.update { it.copy(question = value.take(2000)) } }
    fun filter(id: String?) { ask.update { it.copy(reportId = id) } }
    fun refresh() { viewModelScope.launch { try { repo.refresh(); libraryError.value = null }
        catch (e: Exception) { libraryError.value = "Cannot refresh reports. Saved reports are available offline." } } }
    fun submit() {
        val previous = ask.value
        if (previous.loading || previous.question.isBlank()) return
        val question = previous.question.trim()
        ask.value = previous.copy(text = "", result = null, loading = true, error = null)
        job = viewModelScope.launch {
            try {
                repo.api.answer(AnswerRequest(question, turns.takeLast(6), AnswerFilters(report_ids = listOfNotNull(previous.reportId))))
                    .collect { event -> when (event.name) {
                        "delta" -> ask.update { it.copy(text = it.text + codec.parseToJsonElement(event.data).jsonObject.getValue("text").jsonPrimitive.content) }
                        "complete" -> {
                            val result = codec.decodeFromString<AnswerComplete>(event.data)
                            repo.store.save(Conversation(result.request_id, question, event.data, System.currentTimeMillis()))
                            turns.add(ChatTurn("user", question)); turns.add(ChatTurn("assistant", result.answer_markdown.take(8000)))
                            ask.update { it.copy(text = result.answer_markdown, result = result) }
                        }
                    } }
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) { ask.update { it.copy(error = "We could not finish the answer. Check your connection and retry.") } }
            finally { ask.update { it.copy(loading = false) } }
        }
    }
    fun cancel() { job?.cancel(); ask.update { it.copy(error = "Answer stopped. You can retry.") } }
    fun delete(id: String) { viewModelScope.launch { repo.store.delete(id) } }
    fun clear() { viewModelScope.launch { repo.store.clear(); turns.clear() } }
}
