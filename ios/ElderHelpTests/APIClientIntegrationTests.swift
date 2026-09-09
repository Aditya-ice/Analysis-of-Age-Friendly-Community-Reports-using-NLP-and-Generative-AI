import XCTest

#if !SWIFT_PACKAGE
@testable import ElderHelp

@MainActor
final class APIClientIntegrationTests: XCTestCase {
    func testReportsAndFragmentedAnswerStreamFromMockServer() async throws {
        let client = APIClient(baseURL: try XCTUnwrap(URL(string: "http://localhost:8765")))
        _ = try await client.connect(invite: "mock-invite-code-only")
        let reports = try await client.reports()
        XCTAssertEqual(reports.first?.title, "Age-friendly NYC")

        let stream = await client.answer(AnswerRequest(question: "How can communities help?"))
        var delta = ""
        var completion: AnswerComplete?
        for try await event in stream {
            switch event {
            case .started, .progress:
                break
            case let .delta(text):
                delta += text
            case let .completed(value):
                completion = value
            }
        }

        XCTAssertEqual(delta, "The 2017 report describes safe housing support [S1].")
        XCTAssertEqual(completion?.status, "partial")
        XCTAssertEqual(completion?.citations.first?.id, "S1")
    }
}
#endif
