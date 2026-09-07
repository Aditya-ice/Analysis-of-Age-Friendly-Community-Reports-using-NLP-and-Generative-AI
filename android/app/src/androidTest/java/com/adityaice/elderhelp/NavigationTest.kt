package com.adityaice.elderhelp

import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import org.junit.Rule
import org.junit.Test

class NavigationTest {
    @get:Rule val compose = createAndroidComposeRule<MainActivity>()
    @Test fun askCitationsReportsHistory() {
        compose.onNodeWithText("Your question").performTextInput("How can communities support older adults?")
        compose.onNodeWithText("Ask ElderHelp").performScrollTo().performClick()
        compose.waitUntil(15000) {
            compose.onAllNodesWithText("Supported by report evidence").fetchSemanticsNodes().isNotEmpty()
        }
        compose.onNodeWithText("[S1] Age-friendly NYC, page 12").performScrollTo().performClick()
        compose.onNodeWithText("The plan describes safe housing support.").assertIsDisplayed()
        compose.onNodeWithText("Close source").performClick()
        compose.onNodeWithText("Reports", useUnmergedTree = true).performClick()
        compose.waitUntil(10000) { compose.onAllNodesWithText("Age-friendly NYC").fetchSemanticsNodes().isNotEmpty() }
        compose.onNodeWithText("Age-friendly NYC").assertExists()
        compose.onNodeWithText("History", useUnmergedTree = true).performClick()
        compose.onNodeWithText("How can communities support older adults?").performClick()
        compose.onNodeWithText("Supported by report evidence").assertExists()
        compose.onNodeWithText("Close saved answer").performScrollTo().performClick()
        compose.onNodeWithText("Clear all history").performClick()
        compose.onNodeWithText("Delete all").performClick()
        compose.onNodeWithText("Your completed answers will appear here.").assertExists()
    }
}
