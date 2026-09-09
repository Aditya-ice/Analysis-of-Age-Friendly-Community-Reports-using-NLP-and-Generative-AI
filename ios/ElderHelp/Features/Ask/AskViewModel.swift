import Foundation
import Observation

@MainActor
@Observable
final class AskViewModel {
    var question = ""
    var answerMarkdown = ""
    var citations: [Citation] = []
    var isLoading = false
    var errorMessage: String?
    var selectedCitation: Citation?
    var progress = ""
    var completionStatus = ""
    var missingParts: [String] = []
    var searchHits: [SearchHit] = []
    var followup = false
    var reportID: UUID?
    private var turns: [ChatTurn] = []

    private let apiClient: APIClient
    private var answerTask: Task<Void, Never>?

    init(apiClient: APIClient) {
        self.apiClient = apiClient
    }

    var linkedAnswerMarkdown: String {
        answerMarkdown.replacingOccurrences(
            of: #"\[(S\d+)\]"#,
            with: "[$1](elderhelp://citation/$1)",
            options: .regularExpression
        )
    }

    func submit(
        onComplete: @MainActor @escaping (AnswerComplete) -> Void
    ) {
        let trimmed = question.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, trimmed.count <= 2_000, !isLoading else { return }
        answerTask?.cancel()
        let request = AnswerRequest(question: trimmed, history: followup ? turns : [], filters: .init(reportIDs: reportID.map { [$0] } ?? []))
        answerMarkdown = ""
        citations = []
        errorMessage = nil
        isLoading = true
        searchHits = []; missingParts = []; completionStatus = ""
        progress = "Connecting… The host may need time to wake."
        answerTask = Task {
            do {
                let stream = await apiClient.answer(request)
                for try await event in stream {
                    guard !Task.isCancelled else { return }
                    switch event {
                    case .started:
                        break
                    case let .progress(_, message):
                        progress = message
                    case .delta:
                        break // Display only the completed, verified response.
                    case let .completed(completion):
                        answerMarkdown = completion.answerMarkdown
                        citations = completion.citations
                        completionStatus = completion.status
                        missingParts = completion.missingParts ?? []
                        turns = Array((request.history + [ChatTurn(role: "user", content: trimmed), ChatTurn(role: "assistant", content: String(completion.answerMarkdown.prefix(8000)))]).suffix(6))
                        progress = "Verification complete."
                        onComplete(completion)
                    }
                }
            } catch {
                if !Task.isCancelled {
                    errorMessage = connectionMessage(error)
                }
            }
            if !Task.isCancelled { isLoading = false }
        }
    }

    func cancel() {
        answerTask?.cancel()
        answerTask = nil
        isLoading = false
        errorMessage = "Answer stopped. Nothing unfinished was saved."
        progress = ""
    }

    func newQuestion() {
        cancel(); turns = []; followup = false; question = ""; answerMarkdown = ""
        citations = []; searchHits = []; errorMessage = nil; completionStatus = ""
    }

    func search() {
        guard !isLoading, !question.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        isLoading = true; errorMessage = nil; progress = "Searching approved passages…"
        answerTask = Task {
            do {
                let value = try await apiClient.search(question: question, filters: .init(reportIDs: reportID.map { [$0] } ?? [])).items
                guard !Task.isCancelled else { return }
                searchHits = value
                progress = searchHits.isEmpty ? "No matching passages. Try fewer keywords." : "Keyword results — no generated answer."
            } catch { if !Task.isCancelled { errorMessage = connectionMessage(error) } }
            if !Task.isCancelled { isLoading = false }
        }
    }

    func openCitation(url: URL) -> Bool {
        guard url.scheme == "elderhelp", let identifier = url.pathComponents.last else {
            return false
        }
        selectedCitation = citations.first { $0.id == identifier }
        return selectedCitation != nil
    }
}
