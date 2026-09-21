package com.adityaice.elderhelp

import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import org.junit.Rule
import org.junit.Test

class NavigationTest {
    @get:Rule val compose = createAndroidComposeRule<MainActivity>()
    @Test fun askCitationsReportsHistory() {
        compose.onNodeWithText("Invite code").performTextInput("mock-invite-code-only")
        compose.onNodeWithText("Connect to pilot").performScrollTo().performClick()
        compose.waitUntil(10000) { compose.onAllNodesWithText("Connected. Answers appear after verification.").fetchSemanticsNodes().isNotEmpty() }
        compose.onNodeWithText("Your question").performScrollTo().performTextInput("How can communities support older adults?")
        compose.onNodeWithText("Ask ElderHelp").performScrollTo().performClick()
        compose.waitUntil(15000) {
            compose.onAllNodesWithText("Partial answer — some evidence is missing").fetchSemanticsNodes().isNotEmpty()
        }
        compose.onNodeWithText("[S1] Age-friendly NYC, page 12").performScrollTo().performClick()
        compose.onNodeWithText("The 2017 report describes safe housing support.").assertIsDisplayed()
        compose.onNodeWithText("Close source").performClick()
        compose.onNodeWithText("Reports", useUnmergedTree = true).performClick()
        compose.waitUntil(10000) { compose.onAllNodesWithText("Age-friendly NYC").fetchSemanticsNodes().isNotEmpty() }
        compose.onNodeWithText("Age-friendly NYC").assertExists()
        compose.onNodeWithText("History", useUnmergedTree = true).performClick()
        compose.onNodeWithText("How can communities support older adults?").performClick()
        compose.onNodeWithText("Partial answer — some evidence is missing").assertExists()
        compose.onNodeWithText("Close saved answer").performScrollTo().performClick()
        compose.onNodeWithText("Clear all history").performClick()
        compose.onNodeWithText("Delete all").performClick()
        compose.onNodeWithText("Your completed answers will appear here.").assertExists()
    }
}
