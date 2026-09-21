import XCTest

#if !SWIFT_PACKAGE
import SwiftData
@testable import ElderHelp

@MainActor
final class PersistenceTests: XCTestCase {
    func testPilotKeychainRoundTrip() throws {
        let previous = PilotKeychain.read()
        defer { try? PilotKeychain.save(previous) }
        try PilotKeychain.save("unit-test-pilot-token")
        XCTAssertEqual(PilotKeychain.read(), "unit-test-pilot-token")
        try PilotKeychain.save(nil)
        XCTAssertNil(PilotKeychain.read())
    }

    func testConversationAndCitationRemainOnDevice() throws {
        let container = try ModelContainer(
            for: StoredConversation.self,
            CachedReport.self,
            configurations: ModelConfiguration(isStoredInMemoryOnly: true)
        )
        let context = ModelContext(container)
        let completion = try JSONDecoder().decode(AnswerComplete.self, from: Data(Self.answer.utf8))
        let conversation = StoredConversation(question: "What helps people age in place?", completion: completion)

        context.insert(conversation)
        try context.save()

        let saved = try context.fetch(FetchDescriptor<StoredConversation>())
        XCTAssertEqual(saved.count, 1)
        XCTAssertEqual(saved.first?.citations.first?.pageNumber, 12)

        if let first = saved.first { context.delete(first) }
        try context.save()
        XCTAssertTrue(try context.fetch(FetchDescriptor<StoredConversation>()).isEmpty)
    }

    private static let answer = #"""
    {
      "request_id":"9AA2870C-BF64-4F29-94C2-E4D670A36D2D",
      "answer_markdown":"Safe housing helps [S1].",
      "status":"grounded",
      "citations":[{
        "id":"S1",
        "report_id":"1AA2870C-BF64-4F29-94C2-E4D670A36D2D",
        "report_title":"Age-friendly NYC",
        "publisher":"City of New York",
        "source_url":"https://example.com/report.pdf",
        "publication_date":"2017-01-01",
        "page_number":12,
        "excerpt":"Evidence about safe housing."
      }]
    }
    """#
}
#endif
