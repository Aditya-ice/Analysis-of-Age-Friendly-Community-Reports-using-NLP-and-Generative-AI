import Foundation

public enum APIError: LocalizedError, Sendable {
    case invalidResponse
    case server(Int, String)
    case invalidEvent(String)

    public var errorDescription: String? {
        switch self {
        case .invalidResponse: "The connection ended before the verified answer was complete. Please retry."
        case let .server(code, retry):
            switch code {
            case 401: "Pilot access expired or is invalid. Enter your invite code again. Saved history remains available."
            case 429: "The pilot reached a usage limit. Use keyword search, or retry after \(retry.isEmpty ? "60" : retry) seconds."
            case 413, 422: "Check your question: use 1–2,000 characters and valid report filters."
            default: "The service is unavailable or waking from sleep. Wait a moment and retry."
            }
        case .invalidEvent: "The answer stream was incomplete or unreadable. Nothing was saved. Please retry."
        }
    }
}

public func connectionMessage(_ error: Error) -> String {
    if let api = error as? APIError { return api.localizedDescription }
    if (error as NSError).code == NSURLErrorNotConnectedToInternet {
        return "You are offline. Saved reports and history are available on this device."
    }
    return "The host may be waking up, or the connection failed. Wait a moment and retry."
}

public actor APIClient {
    private let baseURL: URL
    private let session: URLSession
    private let decoder = JSONDecoder()
    private let encoder = JSONEncoder()
    private var token: String?

    public init(baseURL: URL, session: URLSession = .shared, token: String? = nil) {
        self.baseURL = baseURL; self.session = session; self.token = token
    }
    public func setToken(_ value: String?) { token = value }
    private func request(_ path: String, body: Data? = nil) -> URLRequest {
        var value = URLRequest(url: baseURL.appending(path: path), timeoutInterval: 90)
        if let token { value.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
        if let body { value.httpMethod = "POST"; value.httpBody = body; value.setValue("application/json", forHTTPHeaderField: "Content-Type") }
        return value
    }
    private func validate(_ response: URLResponse) throws {
        guard let http = response as? HTTPURLResponse else { throw APIError.invalidResponse }
        guard (200..<300).contains(http.statusCode) else {
            throw APIError.server(http.statusCode, http.value(forHTTPHeaderField: "Retry-After") ?? "")
        }
    }
    private func get<T: Decodable & Sendable>(_ path: String, body: Data? = nil) async throws -> T {
        let (data, response) = try await session.data(for: request(path, body: body))
        try validate(response)
        return try decoder.decode(T.self, from: data)
    }
    public func connect(invite: String) async throws -> PilotSession {
        let value: PilotSession = try await get("/v2/demo/session", body: encoder.encode(["invite_code": invite]))
        token = value.token
        return value
    }
    public func capabilities() async throws -> Capabilities { try await get("/v2/capabilities") }
    public func reports() async throws -> [ReportSummary] {
        var all: [ReportSummary] = []
        while true {
            var components = URLComponents(url: baseURL.appending(path: "/v2/reports"), resolvingAgainstBaseURL: false)!
            components.queryItems = [URLQueryItem(name: "offset", value: String(all.count)), URLQueryItem(name: "limit", value: "100")]
            var value = URLRequest(url: components.url!, timeoutInterval: 90)
            if let token { value.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
            let (data, response) = try await session.data(for: value)
            try validate(response)
            let page = try decoder.decode(ReportList.self, from: data)
            all.append(contentsOf: page.items)
            if all.count >= page.total || page.items.isEmpty { return all }
        }
    }
    public func report(id: UUID) async throws -> ReportDetail { try await get("/v2/reports/\(id.uuidString)") }
    public func search(question: String, filters: AnswerFilters = .init()) async throws -> SearchResults {
        // SearchRequest defaults to keyword mode; this route never silently requests embeddings.
        try await get("/v2/search", body: encoder.encode(AnswerRequest(question: question, filters: filters)))
    }
    public func answer(_ payload: AnswerRequest) -> AsyncThrowingStream<AnswerStreamEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    var value = request("/v2/answers/stream", body: try encoder.encode(payload))
                    value.setValue("text/event-stream", forHTTPHeaderField: "Accept")
                    let (bytes, response) = try await session.bytes(for: value)
                    try validate(response)
                    var parser = SSEParser(); var line = Data(); var started = false; var complete = false; var deltas = ""
                    // Decode complete UTF-8 lines ourselves; preserve blank SSE separators and CRLF.
                    for try await byte in bytes {
                        try Task.checkCancellation()
                        if byte != 10 { line.append(byte); if line.count > 131072 { throw APIError.invalidResponse }; continue }
                        if line.last == 13 { line.removeLast() }
                        guard let text = String(data: line, encoding: .utf8) else { throw APIError.invalidResponse }
                        line.removeAll(keepingCapacity: true)
                        guard let raw = parser.ingest(line: text) else { continue }
                        let event = try decode(raw)
                        if complete { throw APIError.invalidEvent("after completion") }
                        switch event {
                        case .started:
                            guard !started else { throw APIError.invalidEvent("duplicate start") }; started = true
                        case .progress:
                            guard started else { throw APIError.invalidEvent("progress") }
                        case let .delta(text):
                            guard started else { throw APIError.invalidEvent("delta") }; deltas += text
                            guard deltas.count <= 64000 else { throw APIError.invalidResponse }
                        case let .completed(result):
                            guard started, deltas.isEmpty || deltas == result.answerMarkdown,
                                  ["grounded", "partial", "insufficient_evidence", "clarification_required"].contains(result.status),
                                  result.citations.allSatisfy({ $0.spanID != nil && $0.revisionID != nil })
                            else { throw APIError.invalidEvent("complete") }
                            complete = true
                        }
                        continuation.yield(event)
                        if complete { break }
                    }
                    guard complete else { throw APIError.invalidResponse }
                    continuation.finish()
                } catch is CancellationError { continuation.finish() }
                catch { continuation.finish(throwing: error) }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
    private func decode(_ event: RawSSEEvent) throws -> AnswerStreamEvent {
        let data = Data(event.data.utf8)
        switch event.name {
        case "start":
            struct Start: Decodable { let request_id: UUID }
            return .started(try decoder.decode(Start.self, from: data).request_id)
        case "progress":
            struct Progress: Decodable { let stage: String; let message: String }
            let p = try decoder.decode(Progress.self, from: data)
            return .progress(p.stage, p.message)
        case "delta":
            struct Delta: Decodable { let text: String }
            return .delta(try decoder.decode(Delta.self, from: data).text)
        case "complete": return .completed(try decoder.decode(AnswerComplete.self, from: data))
        case "error":
            struct Failure: Decodable { let code: String }
            let failure = try decoder.decode(Failure.self, from: data)
            throw APIError.server(failure.code == "quota_exhausted" ? 429 : 503, "")
        default: throw APIError.invalidEvent(event.name)
        }
    }
}
