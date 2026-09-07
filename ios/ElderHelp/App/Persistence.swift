import Foundation
import SwiftData

@Model
final class StoredConversation {
    @Attribute(.unique) var id: UUID
    var createdAt: Date
    var question: String
    var answerMarkdown: String
    var status: String
    var citationsData: Data

    init(question: String, completion: AnswerComplete) {
        id = completion.requestID
        createdAt = Date()
        self.question = question
        answerMarkdown = completion.answerMarkdown
        status = completion.status
        citationsData = (try? JSONEncoder().encode(completion.citations)) ?? Data()
    }

    var citations: [Citation] {
        (try? JSONDecoder().decode([Citation].self, from: citationsData)) ?? []
    }
}

@Model
final class CachedReport {
    @Attribute(.unique) var id: UUID
    var slug: String
    var title: String
    var publisher: String
    var community: String
    var publicationDate: String?
    var sourceURL: String
    var cachedAt: Date

    init(_ report: ReportSummary) {
        id = report.id
        slug = report.slug
        title = report.title
        publisher = report.publisher
        community = report.community
        publicationDate = report.publicationDate
        sourceURL = report.sourceURL.absoluteString
        cachedAt = Date()
    }

    var summary: ReportSummary? {
        guard let url = URL(string: sourceURL) else { return nil }
        return ReportSummary(
            id: id,
            slug: slug,
            title: title,
            publisher: publisher,
            community: community,
            publicationDate: publicationDate,
            sourceURL: url
        )
    }
}
