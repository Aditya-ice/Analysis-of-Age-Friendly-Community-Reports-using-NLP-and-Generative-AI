import Foundation

public enum APIError: LocalizedError, Sendable {
    case invalidResponse
    case server(Int, String)
    case invalidEvent(String)

    public var errorDescription: String? {
        switch self {
        case .invalidResponse:
            "The server returned an unreadable response."
        case let .server(code, message):
            "The server could not complete the request (\(code)): \(message)"
        case let .invalidEvent(name):
            "The answer stream contained an invalid \(name) event."
        }
    }
}

public actor APIClient {
    private let baseURL: URL
    private let session: URLSession
    private let decoder: JSONDecoder
    private let encoder: JSONEncoder

    public init(baseURL: URL, session: URLSession = .shared) {
        self.baseURL = baseURL
        self.session = session
        decoder = JSONDecoder()
        encoder = JSONEncoder()
    }

    public func reports() async throws -> [ReportSummary] {
        let request = URLRequest(url: baseURL.appending(path: "/v1/reports"))
        let (data, response) = try await session.data(for: request)
        try validate(response: response, data: data)
        return try decoder.decode(ReportList.self, from: data).items
    }

    public func report(id: UUID) async throws -> ReportDetail {
        let request = URLRequest(url: baseURL.appending(path: "/v1/reports/\(id.uuidString)"))
        let (data, response) = try await session.data(for: request)
        try validate(response: response, data: data)
        return try decoder.decode(ReportDetail.self, from: data)
    }

    public func answer(_ payload: AnswerRequest) -> AsyncThrowingStream<AnswerStreamEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    var request = URLRequest(url: baseURL.appending(path: "/v1/answers/stream"))
                    request.httpMethod = "POST"
                    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                    request.setValue("text/event-stream", forHTTPHeaderField: "Accept")
                    request.httpBody = try encoder.encode(payload)
                    let (bytes, response) = try await session.bytes(for: request)
                    guard let http = response as? HTTPURLResponse else {
                        throw APIError.invalidResponse
                    }
                    guard (200..<300).contains(http.statusCode) else {
                        throw APIError.server(http.statusCode, HTTPURLResponse.localizedString(forStatusCode: http.statusCode))
                    }
                    var parser = SSEParser()
                    for try await line in bytes.lines {
                        try Task.checkCancellation()
                        if let event = parser.ingest(line: line) {
                            continuation.yield(try decode(event))
                        }
                    }
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private func decode(_ event: RawSSEEvent) throws -> AnswerStreamEvent {
        let data = Data(event.data.utf8)
        switch event.name {
        case "start":
            struct Start: Decodable { let requestID: UUID; enum CodingKeys: String, CodingKey { case requestID = "request_id" } }
            return .started(try decoder.decode(Start.self, from: data).requestID)
        case "delta":
            struct Delta: Decodable { let text: String }
            return .delta(try decoder.decode(Delta.self, from: data).text)
        case "complete":
            return .completed(try decoder.decode(AnswerComplete.self, from: data))
        case "error":
            struct StreamError: Decodable { let code: String; let message: String }
            let failure = try decoder.decode(StreamError.self, from: data)
            throw APIError.server(503, failure.message)
        default:
            throw APIError.invalidEvent(event.name)
        }
    }

    private func validate(response: URLResponse, data: Data) throws {
        guard let http = response as? HTTPURLResponse else { throw APIError.invalidResponse }
        guard (200..<300).contains(http.statusCode) else {
            let message = String(data: data, encoding: .utf8) ?? "Unknown error"
            throw APIError.server(http.statusCode, message)
        }
    }
}
