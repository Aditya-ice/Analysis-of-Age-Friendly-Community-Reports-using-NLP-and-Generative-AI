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
        history: [ChatTurn],
        onComplete: @MainActor @escaping (AnswerComplete) -> Void
    ) {
        let trimmed = question.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, trimmed.count <= 2_000, !isLoading else { return }
        answerTask?.cancel()
        answerMarkdown = ""
        citations = []
        errorMessage = nil
        isLoading = true
        answerTask = Task {
            do {
                let stream = await apiClient.answer(AnswerRequest(question: trimmed, history: history))
                for try await event in stream {
                    switch event {
                    case .started:
                        break
                    case let .delta(text):
                        answerMarkdown += text
                    case let .completed(completion):
                        answerMarkdown = completion.answerMarkdown
                        citations = completion.citations
                        onComplete(completion)
                    }
                }
            } catch {
                if !Task.isCancelled {
                    errorMessage = error.localizedDescription
                }
            }
            isLoading = false
        }
    }

    func cancel() {
        answerTask?.cancel()
        answerTask = nil
        isLoading = false
    }

    func openCitation(url: URL) -> Bool {
        guard url.scheme == "elderhelp", let identifier = url.pathComponents.last else {
            return false
        }
        selectedCitation = citations.first { $0.id == identifier }
        return selectedCitation != nil
    }
}
