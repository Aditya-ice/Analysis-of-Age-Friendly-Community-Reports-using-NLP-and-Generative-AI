package com.adityaice.elderhelp

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.semantics.heading
import androidx.compose.ui.semantics.liveRegion
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val db = LocalDatabase.get(applicationContext)
        val factory = object : ViewModelProvider.Factory {
            @Suppress("UNCHECKED_CAST")
            override fun <T : ViewModel> create(modelClass: Class<T>): T =
                ElderViewModel(Repository(Api(BuildConfig.API_BASE_URL, tokens = EncryptedPilotTokens(applicationContext, BuildConfig.API_BASE_URL)), db.store())) as T
        }
        setContent { MaterialTheme { ElderApp(viewModel(factory = factory)) } }
    }
}
@OptIn(ExperimentalMaterial3Api::class)
@Composable fun ElderApp(vm: ElderViewModel) {
    var tab by rememberSaveable { mutableIntStateOf(0) }
    val ask by vm.ask.collectAsStateWithLifecycle()
    val reports by vm.reports.collectAsStateWithLifecycle()
    val history by vm.history.collectAsStateWithLifecycle()
    val libraryError by vm.libraryError.collectAsStateWithLifecycle()
    val connected by vm.connected.collectAsStateWithLifecycle()
    val accessMessage by vm.accessMessage.collectAsStateWithLifecycle()
    val detail by vm.detail.collectAsStateWithLifecycle()
    var invite by remember { mutableStateOf("") }
    var selectedReport by remember { mutableStateOf<ReportSummary?>(null) }
    var citation by remember { mutableStateOf<Citation?>(null) }
    var saved by remember { mutableStateOf<Conversation?>(null) }
    var clearConfirmation by remember { mutableStateOf(false) }
    val uri = LocalUriHandler.current
    Scaffold(topBar = { TopAppBar(title = { Text(listOf("ElderHelp", "Reports", "History")[tab]) }) },
        bottomBar = { NavigationBar { listOf("Ask", "Reports", "History").forEachIndexed { index, title ->
            NavigationBarItem(selected = tab == index, onClick = { tab = index },
                icon = { Text(listOf("?", "▤", "◷")[index]) }, label = { Text(title) })
        } } }) { padding ->
        Column(Modifier.padding(padding).fillMaxSize().verticalScroll(rememberScrollState()).padding(20.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)) {
            when (tab) {
                0 -> {
                    Text("Ask about age-friendly communities", style = MaterialTheme.typography.headlineSmall, modifier = Modifier.semantics { heading() })
                    Text("Answers concern historical report findings and appear after verification. Your history stays on this device.")
                    Text("Use non-sensitive research questions. Questions and evidence are sent to Google under its unpaid-service terms.")
                    TextButton(onClick = { uri.openUri("https://ai.google.dev/gemini-api/terms") }) { Text("Google service terms") }
                    Text(if(connected) "Pilot access: session saved" else "Pilot access: invite required")
                    OutlinedTextField(value = invite, onValueChange = { invite = it.take(256) }, label = { Text("Invite code") },
                        visualTransformation = androidx.compose.ui.text.input.PasswordVisualTransformation(), modifier = Modifier.fillMaxWidth())
                    Button(onClick = { vm.connect(invite); invite = "" }, enabled = invite.length >= 12) { Text("Connect to pilot") }
                    if(connected) TextButton(onClick = vm::forget) { Text("Forget access token") }
                    accessMessage?.let { Text(it) }
                    ask.reportId?.let { id ->
                        Text("Report: ${reports.firstOrNull { it.id == id }?.title ?: id}")
                        TextButton(onClick = { vm.filter(null) }) { Text("Search all reports") }
                    }
                    OutlinedTextField(value = ask.question, onValueChange = vm::question,
                        enabled = !ask.loading, label = { Text("Your question") },
                        supportingText = { Text("${ask.question.length} / 2,000 characters") },
                        modifier = Modifier.fillMaxWidth(), minLines = 3)
                    Row(verticalAlignment = androidx.compose.ui.Alignment.CenterVertically) {
                        Checkbox(checked = ask.followup, onCheckedChange = vm::followup, enabled = !ask.loading)
                        Text("Follow up on this conversation")
                    }
                    Text("New questions do not use saved history. Follow-ups use only this active conversation.")
                    Button(onClick = vm::submit, enabled = ask.question.isNotBlank() && !ask.loading,
                        modifier = Modifier.fillMaxWidth().heightIn(min = 52.dp)) { Text(if (ask.error != null) "Retry answer" else "Ask ElderHelp") }
                    Button(onClick = vm::search, enabled = !ask.loading && ask.question.isNotBlank()) { Text("Search passages") }
                    TextButton(onClick = vm::newQuestion, enabled = !ask.loading) { Text("New question") }
                    if(ask.progress.isNotEmpty()) Text(ask.progress, modifier = Modifier.semantics { liveRegion = androidx.compose.ui.semantics.LiveRegionMode.Polite })
                    if (ask.loading) {
                        Button(onClick = vm::cancel) { Text("Stop answer") }
                    }
                    ask.error?.let { Text(it) }
                    ask.hits.forEach { hit ->
                        Card { Column(Modifier.padding(16.dp)) { Text(hit.citation.excerpt)
                            TextButton(onClick = { citation = hit.citation }) { Text("Inspect ${hit.citation.report_title}, page ${hit.citation.page_number}") }
                        } }
                    }
                    if (ask.text.isNotEmpty()) AnswerContent(ask.text, ask.result, { citation = it })
                }
                1 -> {
                    Button(onClick = vm::refresh) { Text("Refresh reports") }
                    libraryError?.let { Text(it) }
                    if (reports.isEmpty()) Text("No saved reports yet. Connect to the service and refresh.")
                    reports.forEach { report ->
                        Card(Modifier.fillMaxWidth()) { Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                            Text(report.title, style = MaterialTheme.typography.titleLarge)
                            Text("${report.publisher} · ${report.community}")
                            Text(report.publication_date ?: "Publication date unavailable")
                            TextButton(onClick = { selectedReport = report; vm.loadDetail(report.id) }) { Text("Report details") }
                            Button(onClick = { vm.filter(report.id); tab = 0 }) { Text("Ask about this report") }
                            TextButton(onClick = { uri.openUri(report.source_url) }) { Text("Open publisher source") }
                        } }
                    }
                }
                2 -> {
                    Text("Saved on this device. Available without an internet connection.")
                    if (history.isEmpty()) Text("Your completed answers will appear here.")
                    if (history.isNotEmpty()) TextButton(onClick = { clearConfirmation = true }) { Text("Clear all history") }
                    history.forEach { conversation ->
                        Card(Modifier.fillMaxWidth()) { Column(Modifier.padding(16.dp)) {
                            TextButton(onClick = { saved = conversation }) { Text(conversation.question) }
                            TextButton(onClick = { vm.delete(conversation.id) }) { Text("Delete conversation") }
                        } }
                    }
                }
            }
        }
    }
    selectedReport?.let { report ->
        ModalBottomSheet(onDismissRequest = { selectedReport = null }) {
            Column(Modifier.verticalScroll(rememberScrollState()).padding(24.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Text(report.title, style = MaterialTheme.typography.titleLarge)
                Text("${report.publisher} · ${report.publication_date ?: "Date unknown"}")
                detail?.takeIf { it.id == report.id }?.let { d ->
                    Text(d.description ?: "")
                    d.suggested_questions.forEach { q -> TextButton(onClick = { vm.question(q); vm.filter(report.id); selectedReport = null; tab = 0 }) { Text(q) } }
                }
                Button(onClick = { selectedReport = null }) { Text("Close report") }
            }
        }
    }
    saved?.let { conversation ->
        ModalBottomSheet(onDismissRequest = { saved = null }) {
            Column(Modifier.verticalScroll(rememberScrollState()).padding(24.dp)) {
                Text(conversation.question, style = MaterialTheme.typography.titleLarge)
                val result = codec.decodeFromString<AnswerComplete>(conversation.answer)
                AnswerContent(result.answer_markdown, result) { citation = it }
                TextButton(onClick = { saved = null }) { Text("Close saved answer") }
            }
        }
    }
    citation?.let { source ->
        ModalBottomSheet(onDismissRequest = { citation = null }) {
            Column(Modifier.verticalScroll(rememberScrollState()).padding(24.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                Text("Source ${source.id}", style = MaterialTheme.typography.headlineSmall)
                Text(source.report_title, style = MaterialTheme.typography.titleLarge)
                Text("${source.publisher} · Page ${source.page_number}")
                Text(source.publication_date ?: "Publication date unavailable")
                source.page_label?.let { Text("Printed page label: $it") }
                Text(source.excerpt)
                TextButton(onClick = { uri.openUri(source.source_url) }) { Text("Open publisher source") }
                Button(onClick = { citation = null }) { Text("Close source") }
            }
        }
    }
    if (clearConfirmation) AlertDialog(onDismissRequest = { clearConfirmation = false },
        title = { Text("Delete all local history?") }, text = { Text("These conversations cannot be recovered.") },
        confirmButton = { TextButton(onClick = { vm.clear(); clearConfirmation = false }) { Text("Delete all") } },
        dismissButton = { TextButton(onClick = { clearConfirmation = false }) { Text("Keep history") } })
}
@Composable private fun AnswerContent(text: String, result: AnswerComplete?, onCitation: (Citation) -> Unit) {
    Text("Answer", style = MaterialTheme.typography.titleLarge, modifier = Modifier.semantics { heading() })
    // Each citation marker is an individually accessible link within the answer text.
    val links = androidx.compose.ui.text.buildAnnotatedString {
        var offset = 0
        Regex("\\[(S[0-9]+)\\]").findAll(text).forEach { match ->
            append(text.substring(offset, match.range.first))
            val source = result?.citations?.firstOrNull { it.id == match.groupValues[1] }
            if (source != null) {
                pushLink(androidx.compose.ui.text.LinkAnnotation.Clickable(source.id,
                    androidx.compose.ui.text.TextLinkStyles(style = androidx.compose.ui.text.SpanStyle(textDecoration = androidx.compose.ui.text.style.TextDecoration.Underline)),
                    { onCitation(source) }))
                append(match.value); pop()
            } else append(match.value)
            offset = match.range.last + 1
        }
        append(text.substring(offset))
    }
    Text(links)
    result?.let {
        Text(when(it.status) { "grounded" -> "Supported by report evidence"; "partial" -> "Partial answer — some evidence is missing"; "clarification_required" -> "Clarification needed"; else -> "Insufficient evidence in these reports" })
        if(it.missing_parts.isNotEmpty()) Text("Missing evidence: " + it.missing_parts.joinToString("; "))
        it.citations.forEach { source ->
            TextButton(onClick = { onCitation(source) }, modifier = Modifier.heightIn(min = 48.dp)) {
                Text("[${source.id}] ${source.report_title}, page ${source.page_number}")
            }
        }
    }
}
