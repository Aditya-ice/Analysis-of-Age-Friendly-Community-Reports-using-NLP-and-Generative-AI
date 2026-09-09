import XCTest

@MainActor
final class ElderHelpUITests: XCTestCase {
    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    func testAskReportsHistoryAndCitations() {
        let app = XCUIApplication()
        app.launchEnvironment["ELDERHELP_API_BASE_URL"] = "http://localhost:8765"
        app.launch()

        XCTAssertTrue(app.navigationBars["ElderHelp"].waitForExistence(timeout: 5))
        app.buttons.matching(NSPredicate(format: "label BEGINSWITH 'Pilot access'")).firstMatch.tap()
        let invite = app.secureTextFields["Invite code"]
        XCTAssertTrue(invite.waitForExistence(timeout: 10))
        invite.tap(); invite.typeText("mock-invite-code-only")
        app.buttons["Connect to pilot"].tap()
        let status = app.staticTexts["Pilot status"]
        XCTAssertTrue(status.waitForExistence(timeout: 15), app.debugDescription)
        XCTAssertEqual(status.label, "Connected. Answers appear after verification.", app.debugDescription)
        app.swipeUp()
        let editor = app.textViews["Your question"]
        editor.tap()
        editor.typeText("How can communities support older adults?")
        app.buttons["Ask ElderHelp"].tap()
        XCTAssertTrue(app.keyboards.firstMatch.waitForNonExistence(timeout: 15))

        app.swipeUp()
        XCTAssertTrue(app.staticTexts["Sources"].waitForExistence(timeout: 15))
        let sourceButton = app.buttons.matching(
            NSPredicate(format: "label CONTAINS 'Age-friendly NYC'")
        ).firstMatch
        XCTAssertTrue(sourceButton.waitForExistence(timeout: 15))
        sourceButton.tap()
        XCTAssertTrue(app.navigationBars["Source S1"].waitForExistence(timeout: 15))
        app.buttons["Done"].tap()
        XCTAssertTrue(app.navigationBars["Source S1"].waitForNonExistence(timeout: 15))

        let reportsTab = app.tabBars.buttons["Reports"]
        XCTAssertTrue(reportsTab.waitForExistence(timeout: 15))
        let reportsTabReady = XCTNSPredicateExpectation(
            predicate: NSPredicate(format: "isHittable == true"),
            object: reportsTab
        )
        XCTAssertEqual(XCTWaiter.wait(for: [reportsTabReady], timeout: 15), .completed)
        reportsTab.tap()
        XCTAssertTrue(app.navigationBars["Reports"].waitForExistence(timeout: 15))
        XCTAssertTrue(app.staticTexts["Age-friendly NYC"].waitForExistence(timeout: 15))

        app.tabBars.buttons["History"].tap()
        XCTAssertTrue(app.navigationBars["History"].waitForExistence(timeout: 15))
        XCTAssertTrue(app.staticTexts["How can communities support older adults?"].waitForExistence(timeout: 15))
    }
}
