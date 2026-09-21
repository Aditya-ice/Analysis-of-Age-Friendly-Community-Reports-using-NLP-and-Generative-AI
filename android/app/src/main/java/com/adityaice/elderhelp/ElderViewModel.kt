package com.adityaice.elderhelp

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import kotlinx.serialization.json.*

data class AskState(val question: String = "", val text: String = "", val result: AnswerComplete? = null,
    val loading: Boolean = false, val error: String? = null, val reportId: String? = null,
    val progress: String = "", val followup: Boolean = false, val hits: List<SearchHit> = emptyList())
class ElderViewModel(private val repo: Repository) : ViewModel() {
    private val mutableAsk = MutableStateFlow(AskState())
    val ask = mutableAsk.asStateFlow()
    val libraryError = MutableStateFlow<String?>(null)
    val accessMessage = MutableStateFlow<String?>(null)
    val connected = MutableStateFlow(repo.api.hasAccess())
    val history = repo.store.history().stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())
    val reports = repo.store.reports().map { rows -> rows.map { codec.decodeFromString<ReportSummary>(it.payload) } }
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5000), emptyList())
    val detail = MutableStateFlow<ReportDetail?>(null)
    private var job: Job? = null
    private var turns = listOf<ChatTurn>()
    init { if(connected.value) refresh() }
    fun question(value: String) { if (!ask.value.loading) mutableAsk.update { it.copy(question = value.take(2000)) } }
    fun filter(id: String?) { if(!ask.value.loading) mutableAsk.update { it.copy(reportId = id) } }
    fun followup(value: Boolean) { if(!ask.value.loading) mutableAsk.update { it.copy(followup = value) } }
    fun connect(code: String) { viewModelScope.launch {
        try { repo.api.connect(code); connected.value = true; refresh()
            accessMessage.value = if(repo.api.capabilities().generation_available) "Connected. Answers appear after verification." else "Answers unavailable. Use Search passages."
        } catch(e: Exception) { accessMessage.value = connectionMessage(e) }
    } }
    fun forget() { cancel(); repo.api.forget(); connected.value = false; accessMessage.value = "Access token removed. Saved history remains available." }
    fun refresh() { viewModelScope.launch { try { repo.refresh(); libraryError.value = null }
        catch (e: Exception) { libraryError.value = connectionMessage(e) + " Showing saved reports." } } }
    fun loadDetail(id: String) { detail.value = null; viewModelScope.launch { try { detail.value = repo.api.report(id) } catch (_: Exception) { libraryError.value = "Detailed information is unavailable. Saved metadata remains available." } } }
    fun submit() {
        val previous = ask.value
        if (previous.loading || previous.question.isBlank()) return
        val question = previous.question.trim()
        val payload = AnswerRequest(question, if(previous.followup) turns.takeLast(6) else emptyList(), AnswerFilters(report_ids = listOfNotNull(previous.reportId)))
        mutableAsk.value = previous.copy(text = "", result = null, loading = true, error = null, hits = emptyList(), progress = "Connecting… The host may need time to wake.")
        job = viewModelScope.launch {
            try {
                repo.api.answer(payload).collect { event -> currentCoroutineContext().ensureActive(); when (event.name) {
                    "progress" -> mutableAsk.update { it.copy(progress = codec.parseToJsonElement(event.data).jsonObject.getValue("message").jsonPrimitive.content) }
                    "complete" -> {
                        val result = codec.decodeFromString<AnswerComplete>(event.data)
                        repo.store.save(Conversation(result.request_id, question, event.data, System.currentTimeMillis()))
                        turns = (payload.history + listOf(ChatTurn("user", question), ChatTurn("assistant", result.answer_markdown.take(8000)))).takeLast(6)
                        mutableAsk.update { it.copy(text = result.answer_markdown, result = result, progress = "Verification complete. Saved on this device.") }
                    }
                } }
            } catch (e: CancellationException) { throw e }
            catch (e: Exception) { mutableAsk.update { it.copy(error = connectionMessage(e)) } }
            finally { if(currentCoroutineContext().isActive) mutableAsk.update { it.copy(loading = false) } }
        }
    }
    fun search() {
        val previous = ask.value
        if(previous.loading || previous.question.isBlank()) return
        mutableAsk.update { it.copy(loading = true, error = null, progress = "Searching approved passages…") }
        job = viewModelScope.launch {
            try {
                val result = repo.api.search(AnswerRequest(previous.question.trim(), filters = AnswerFilters(report_ids = listOfNotNull(previous.reportId))))
                mutableAsk.update { it.copy(hits = result.items, progress = if(result.items.isEmpty()) "No matching passages. Try fewer keywords." else "Keyword results — no generated answer.") }
            } catch(e: CancellationException) { throw e }
            catch(e: Exception) { mutableAsk.update { it.copy(error = connectionMessage(e)) } }
            finally { if(currentCoroutineContext().isActive) mutableAsk.update { it.copy(loading = false) } }
        }
    }
    fun cancel() { job?.cancel(); mutableAsk.update { it.copy(loading = false, error = "Answer stopped. Nothing unfinished was saved.", progress = "") } }
    fun newQuestion() { cancel(); turns = emptyList(); mutableAsk.value = AskState() }
    fun delete(id: String) { viewModelScope.launch { repo.store.delete(id) } }
    fun clear() { viewModelScope.launch { repo.store.clear(); turns = emptyList() } }
}
