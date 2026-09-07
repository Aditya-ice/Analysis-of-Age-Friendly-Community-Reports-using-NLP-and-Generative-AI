import XCTest

@MainActor
final class ElderHelpUITests: XCTestCase {
    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    func testAskReportsHistoryAndCitations() {
        let app = XCUIApplication()
        app.launchEnvironment["ELDERHELP_API_BASE_URL"] = "http://127.0.0.1:8765"
        app.launch()

        XCTAssertTrue(app.navigationBars["ElderHelp"].waitForExistence(timeout: 5))
        let editor = app.textViews["Your question"]
        editor.tap()
        editor.typeText("How can communities support older adults?")
        app.buttons["Ask ElderHelp"].tap()

        XCTAssertTrue(app.staticTexts["Sources"].waitForExistence(timeout: 10))
        app.buttons.matching(NSPredicate(format: "label CONTAINS 'Age-friendly NYC'")).firstMatch.tap()
        XCTAssertTrue(app.navigationBars["Source S1"].waitForExistence(timeout: 5))
        app.buttons["Done"].tap()

        app.tabBars.buttons["Reports"].tap()
        XCTAssertTrue(app.navigationBars["Reports"].waitForExistence(timeout: 5))
        XCTAssertTrue(app.staticTexts["Age-friendly NYC"].waitForExistence(timeout: 5))

        app.tabBars.buttons["History"].tap()
        XCTAssertTrue(app.navigationBars["History"].waitForExistence(timeout: 5))
        XCTAssertTrue(app.staticTexts["How can communities support older adults?"].waitForExistence(timeout: 5))
    }
}
