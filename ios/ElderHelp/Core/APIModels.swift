import Foundation

public struct ReportSummary: Codable, Identifiable, Hashable, Sendable {
    public let id: UUID
    public let slug: String
    public let title: String
    public let publisher: String
    public let community: String
    public let publicationDate: String?
    public let sourceURL: URL

    enum CodingKeys: String, CodingKey {
        case id, slug, title, publisher, community
        case publicationDate = "publication_date"
        case sourceURL = "source_url"
    }
}

public struct ReportDetail: Codable, Identifiable, Sendable {
    public let id: UUID
    public let slug: String
    public let title: String
    public let publisher: String
    public let community: String
    public let publicationDate: String?
    public let sourceURL: URL
    public let description: String?
    public let suggestedQuestions: [String]
    public let pageCount: Int

    enum CodingKeys: String, CodingKey {
        case id, slug, title, publisher, community, description
        case publicationDate = "publication_date"
        case sourceURL = "source_url"
        case suggestedQuestions = "suggested_questions"
        case pageCount = "page_count"
    }
}

struct ReportList: Codable, Sendable {
    let items: [ReportSummary]
    let total: Int
}

public struct Citation: Codable, Identifiable, Hashable, Sendable {
    public let id: String
    public let reportID: UUID
    public let reportTitle: String
    public let publisher: String
    public let sourceURL: URL
    public let publicationDate: String?
    public let pageNumber: Int
    public let excerpt: String
    public let revisionID: UUID?
    public let spanID: UUID?
    public let pageLabel: String?

    enum CodingKeys: String, CodingKey {
        case id, publisher, excerpt
        case reportID = "report_id"
        case reportTitle = "report_title"
        case sourceURL = "source_url"
        case publicationDate = "publication_date"
        case pageNumber = "page_number"
        case revisionID = "revision_id"
        case spanID = "span_id"
        case pageLabel = "page_label"
    }
}

public struct ChatTurn: Codable, Hashable, Sendable {
    public let role: String
    public let content: String

    public init(role: String, content: String) {
        self.role = role
        self.content = content
    }
}

public struct AnswerFilters: Codable, Sendable {
    public var reportIDs: [UUID] = []
    public var community: String?
    public var yearFrom: Int?
    public var yearTo: Int?

    enum CodingKeys: String, CodingKey {
        case community
        case reportIDs = "report_ids"
        case yearFrom = "year_from"
        case yearTo = "year_to"
    }

    public init(reportIDs: [UUID] = [], community: String? = nil, yearFrom: Int? = nil, yearTo: Int? = nil) {
        self.reportIDs = reportIDs
        self.community = community
        self.yearFrom = yearFrom
        self.yearTo = yearTo
    }
}

public struct AnswerRequest: Codable, Sendable {
    public let question: String
    public let history: [ChatTurn]
    public let filters: AnswerFilters

    public init(question: String, history: [ChatTurn] = [], filters: AnswerFilters = .init()) {
        self.question = question
        self.history = Array(history.suffix(6))
        self.filters = filters
    }
}

public struct AnswerComplete: Codable, Sendable {
    public let requestID: UUID
    public let answerMarkdown: String
    public let status: String
    public let citations: [Citation]
    public let missingParts: [String]?
    public let corpusGeneration: UUID?

    enum CodingKeys: String, CodingKey {
        case status, citations
        case requestID = "request_id"
        case missingParts = "missing_parts"
        case corpusGeneration = "corpus_generation"
        case answerMarkdown = "answer_markdown"
    }
}

public enum AnswerStreamEvent: Sendable {
    case started(UUID)
    case progress(String, String)
    case delta(String)
    case completed(AnswerComplete)
}

public struct PilotSession: Codable, Sendable {
    public let token: String
    public let expiresAt: Int
    enum CodingKeys: String, CodingKey { case token; case expiresAt = "expires_at" }
}
public struct Capabilities: Codable, Sendable {
    public let generationAvailable: Bool
    public let searchAvailable: Bool
    public let reason: String?
    enum CodingKeys: String, CodingKey {
        case generationAvailable = "generation_available"
        case searchAvailable = "search_available"
        case reason
    }
}
public struct SearchHit: Codable, Sendable { public let citation: Citation; public let score: Double }
public struct SearchResults: Codable, Sendable { public let items: [SearchHit]; public let mode: String }
public func statusLabel(_ status: String) -> String {
    switch status {
    case "grounded": "Supported by report evidence"
    case "partial": "Partial answer — some evidence is missing"
    case "clarification_required": "Clarification needed"
    default: "Insufficient evidence in these reports"
    }
}
