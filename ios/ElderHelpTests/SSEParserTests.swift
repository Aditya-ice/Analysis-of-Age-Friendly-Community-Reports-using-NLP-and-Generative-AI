import XCTest
#if SWIFT_PACKAGE
@testable import ElderHelpCore
#else
@testable import ElderHelp
#endif

final class SSEParserTests: XCTestCase {
    func testParsesNamedEventAfterBlankLine() {
        var parser = SSEParser()
        XCTAssertNil(parser.ingest(line: "event: delta"))
        XCTAssertNil(parser.ingest(line: "data: {\"text\":\"Hello\"}"))
        XCTAssertEqual(
            parser.ingest(line: ""),
            RawSSEEvent(name: "delta", data: "{\"text\":\"Hello\"}")
        )
    }

    func testJoinsMultipleDataLinesAndIgnoresComments() {
        var parser = SSEParser()
        XCTAssertNil(parser.ingest(line: ": keep alive"))
        XCTAssertNil(parser.ingest(line: "event: complete"))
        XCTAssertNil(parser.ingest(line: "data: first"))
        XCTAssertNil(parser.ingest(line: "data: second"))
        XCTAssertEqual(
            parser.ingest(line: ""),
            RawSSEEvent(name: "complete", data: "first\nsecond")
        )
    }

    func testResetsEventNameAfterDispatch() {
        var parser = SSEParser()
        _ = parser.ingest(line: "event: start")
        _ = parser.ingest(line: "data: {}")
        _ = parser.ingest(line: "")
        _ = parser.ingest(line: "data: next")
        XCTAssertEqual(parser.ingest(line: ""), RawSSEEvent(name: "message", data: "next"))
    }
}
