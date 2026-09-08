import XCTest
#if SWIFT_PACKAGE
@testable import ElderHelpCore
#else
@testable import ElderHelp
#endif

@MainActor
final class APIModelsTests: XCTestCase {
    func testDecodesCompleteEventContract() throws {
        let json = #"""
        {
          "request_id":"9AA2870C-BF64-4F29-94C2-E4D670A36D2D",
          "answer_markdown":"Housing is discussed [S1].",
          "status":"grounded",
          "citations":[{
            "id":"S1",
            "report_id":"1AA2870C-BF64-4F29-94C2-E4D670A36D2D",
            "report_title":"Age-friendly NYC",
            "publisher":"City of New York",
            "source_url":"https://example.com/report.pdf",
            "publication_date":"2017-01-01",
            "page_number":41,
            "excerpt":"Affordable housing evidence."
          }]
        }
        """#
        let result = try JSONDecoder().decode(AnswerComplete.self, from: Data(json.utf8))
        XCTAssertEqual(result.status, "grounded")
        XCTAssertEqual(result.citations.first?.pageNumber, 41)
        XCTAssertEqual(result.citations.first?.id, "S1")
    }

    func testSharedPartialCitationFixture() throws {
        #if SWIFT_PACKAGE
        let bundle = Bundle.module
        #else
        let bundle = Bundle(for: Self.self)
        #endif
        let path = try XCTUnwrap(bundle.url(forResource: "complete-v2", withExtension: "json", subdirectory: "Fixtures") ?? bundle.url(forResource: "complete-v2", withExtension: "json"))
        let result = try JSONDecoder().decode(AnswerComplete.self, from: Data(contentsOf: path))
        XCTAssertEqual(result.status, "partial")
        XCTAssertEqual(result.citations.first?.publicationDate, "2017")
        XCTAssertNotNil(result.citations.first?.spanID)
        XCTAssertEqual(result.missingParts?.count, 1)
    }

    func testRequestKeepsOnlySixMostRecentTurns() throws {
        let history = (0..<8).map { ChatTurn(role: "user", content: "Question \($0)") }
        let request = AnswerRequest(question: "Latest", history: history)
        let object = try JSONSerialization.jsonObject(with: JSONEncoder().encode(request)) as? [String: Any]
        let encodedHistory = object?["history"] as? [[String: Any]]
        XCTAssertEqual(encodedHistory?.count, 6)
        XCTAssertEqual(encodedHistory?.first?["content"] as? String, "Question 2")
    }
}
