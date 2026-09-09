import SwiftData
import SwiftUI

struct AskView: View {
    @State private var viewModel: AskViewModel
    @FocusState private var questionIsFocused: Bool
    @FocusState private var inviteIsFocused: Bool
    @State private var accessExpanded = false
    @Environment(\.modelContext) private var modelContext
    @Query(sort: \CachedReport.title) private var cachedReports: [CachedReport]
    @State private var pilot: PilotAccess
    init(apiClient: APIClient, reportID: UUID? = nil, question: String = "") {
        let model = AskViewModel(apiClient: apiClient)
        model.reportID = reportID; model.question = question
        _viewModel = State(initialValue: model)
        _pilot = State(initialValue: PilotAccess(client: apiClient))
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                Text("Ask about age-friendly communities")
                    .font(.title.bold())
                    .accessibilityAddTraits(.isHeader)
                Text("Answers concern historical findings in approved reports. They appear after their supporting evidence is checked.")
                    .font(.body)
                    .foregroundStyle(.secondary)

                Text("Use non-sensitive research questions. Questions and evidence are sent to Google under its unpaid-service terms.")
                    .font(.footnote)
                Link("Google service terms", destination: URL(string: "https://ai.google.dev/gemini-api/terms")!)
                DisclosureGroup(pilot.connected ? "Pilot access — session saved" : "Pilot access — invite required", isExpanded: $accessExpanded) {
                    SecureField("Invite code", text: $pilot.invite).textInputAutocapitalization(.never).autocorrectionDisabled()
                        .accessibilityIdentifier("Invite code")
                        .focused($inviteIsFocused)
                    Button("Connect to pilot") {
                        questionIsFocused = false; inviteIsFocused = false
                        Task { await pilot.connect(); if pilot.connected { accessExpanded = false } }
                    }
                        .buttonStyle(.borderedProminent).disabled(pilot.busy)
                    if pilot.connected { Button("Forget access token") { viewModel.cancel(); Task { await pilot.disconnect() } } }
                }
                if let message = pilot.message { Text(message).font(.footnote).accessibilityLabel(message).accessibilityIdentifier("Pilot status") }
                TextEditor(text: $viewModel.question)
                    .focused($questionIsFocused)
                    .disabled(viewModel.isLoading)
                    .frame(minHeight: 110)
                    .padding(8)
                    .background(.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 12))
                    .accessibilityLabel("Your question")
                    .accessibilityHint("Enter a question of up to 2,000 characters")

                Picker("Report", selection: $viewModel.reportID) {
                    Text("All approved reports").tag(nil as UUID?)
                    ForEach(cachedReports) { Text($0.title).tag(Optional($0.id)) }
                }.disabled(viewModel.isLoading)
                Toggle("Follow up on this conversation", isOn: $viewModel.followup).disabled(viewModel.isLoading)
                Text("New questions use no saved history. Enable follow-up to use only this active conversation.").font(.footnote)
                HStack {
                    Button {
                        questionIsFocused = false
                        let submitted = viewModel.question.trimmingCharacters(in: .whitespacesAndNewlines)
                        viewModel.submit { completion in
                            modelContext.insert(StoredConversation(question: submitted, completion: completion))
                        }
                    } label: {
                        Label(viewModel.errorMessage == nil ? "Ask ElderHelp" : "Retry answer", systemImage: "arrow.up.circle.fill")
                            .frame(maxWidth: .infinity, minHeight: 50)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(viewModel.question.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || viewModel.question.count > 2_000 || viewModel.isLoading)

                    if viewModel.isLoading {
                        Button("Stop", role: .cancel) { viewModel.cancel() }
                            .buttonStyle(.bordered)
                            .frame(minHeight: 50)
                    }
                }

                Button("Search passages") { questionIsFocused = false; viewModel.search() }
                    .buttonStyle(.bordered).disabled(viewModel.isLoading || viewModel.question.isEmpty || viewModel.question.count > 2_000)
                Button("New question") { viewModel.newQuestion() }.disabled(viewModel.isLoading)
                if !viewModel.progress.isEmpty { Text(viewModel.progress).accessibilityLabel(viewModel.progress) }
                ForEach(viewModel.searchHits, id: \.citation.spanID) { hit in
                    VStack(alignment: .leading) {
                        Text(hit.citation.excerpt)
                        Button("Inspect \(hit.citation.reportTitle), page \(hit.citation.pageNumber)") { viewModel.selectedCitation = hit.citation }
                            .frame(minHeight: 50)
                    }
                }

                if let error = viewModel.errorMessage {
                    Label(error, systemImage: "exclamationmark.triangle")
                        .foregroundStyle(.red)
                        .accessibilityLabel("Error: \(error)")
                }

                if !viewModel.answerMarkdown.isEmpty {
                    answerSection
                }
            }
            .padding()
        }
        .navigationTitle("ElderHelp")
        .sheet(item: $viewModel.selectedCitation) { CitationSheet(citation: $0) }
    }

    private var answerSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(statusLabel(viewModel.completionStatus)).font(.headline).accessibilityAddTraits(.isHeader)
            if !viewModel.missingParts.isEmpty { Text("Missing evidence: " + viewModel.missingParts.joined(separator: "; ")) }
            Text("Answer").font(.title2.bold()).accessibilityAddTraits(.isHeader)
            Text((try? AttributedString(markdown: viewModel.linkedAnswerMarkdown)) ?? AttributedString(viewModel.answerMarkdown))
                .textSelection(.enabled)
                .environment(\.openURL, OpenURLAction { url in
                    viewModel.openCitation(url: url) ? .handled : .systemAction
                })
            if !viewModel.citations.isEmpty {
                Text("Sources").font(.headline).accessibilityAddTraits(.isHeader)
                ForEach(viewModel.citations) { citation in
                    Button {
                        viewModel.selectedCitation = citation
                    } label: {
                        HStack(alignment: .top) {
                            Text("[\(citation.id)]").fontWeight(.semibold)
                            VStack(alignment: .leading) {
                                Text(citation.reportTitle).multilineTextAlignment(.leading)
                                Text("Page \(citation.pageNumber) · \(citation.publisher)")
                                    .font(.subheadline).foregroundStyle(.secondary)
                            }
                            Spacer()
                            Image(systemName: "chevron.right")
                        }
                        .frame(minHeight: 50)
                    }
                    .buttonStyle(.plain)
                    .accessibilityHint("Shows the report excerpt")
                }
            }
        }
        .padding()
        .background(.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 16))
    }
}
