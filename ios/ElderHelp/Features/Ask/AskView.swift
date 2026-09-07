import SwiftData
import SwiftUI

struct AskView: View {
    @State private var viewModel: AskViewModel
    @FocusState private var questionIsFocused: Bool
    @Environment(\.modelContext) private var modelContext
    @Query(sort: \StoredConversation.createdAt, order: .reverse) private var history: [StoredConversation]
    init(apiClient: APIClient) {
        _viewModel = State(initialValue: AskViewModel(apiClient: apiClient))
    }

    private var recentTurns: [ChatTurn] {
        Array(history.prefix(3).reversed()).flatMap {
            [ChatTurn(role: "user", content: $0.question), ChatTurn(role: "assistant", content: $0.answerMarkdown)]
        }
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                Text("Ask about age-friendly communities")
                    .font(.title.bold())
                    .accessibilityAddTraits(.isHeader)
                Text("Answers use approved reports and include the pages that support them.")
                    .font(.body)
                    .foregroundStyle(.secondary)

                TextEditor(text: $viewModel.question)
                    .focused($questionIsFocused)
                    .frame(minHeight: 110)
                    .padding(8)
                    .background(.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 12))
                    .accessibilityLabel("Your question")
                    .accessibilityHint("Enter a question of up to 2,000 characters")

                HStack {
                    Button {
                        questionIsFocused = false
                        viewModel.submit(history: recentTurns) { completion in
                            modelContext.insert(StoredConversation(question: viewModel.question, completion: completion))
                        }
                    } label: {
                        Label("Ask ElderHelp", systemImage: "arrow.up.circle.fill")
                            .frame(maxWidth: .infinity, minHeight: 50)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(viewModel.question.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || viewModel.isLoading)

                    if viewModel.isLoading {
                        Button("Stop", role: .cancel) { viewModel.cancel() }
                            .buttonStyle(.bordered)
                            .frame(minHeight: 50)
                    }
                }

                if viewModel.isLoading {
                    ProgressView("Finding evidence and preparing an answer")
                        .accessibilityLabel("Preparing answer")
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
