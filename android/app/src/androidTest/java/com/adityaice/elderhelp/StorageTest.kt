package com.adityaice.elderhelp

import androidx.room.Room
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import org.junit.Assert.*
import org.junit.Test

class StorageTest {
    @Test fun historyAndReportsPersistOfflineAndDelete() = runBlocking {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val name = "storage-test-${java.util.UUID.randomUUID()}.db"
        var db = Room.databaseBuilder(context, LocalDatabase::class.java, name).build()
        try {
            db.store().save(Conversation("answer", "Question", "{}", 1))
            db.store().replaceReports(listOf(CachedReport("report", "{}")))
            db.close()
            db = Room.databaseBuilder(context, LocalDatabase::class.java, name).build()
            assertEquals("Question", db.store().history().first().single().question)
            assertEquals("report", db.store().reports().first().single().id)
            db.store().delete("answer")
            assertTrue(db.store().history().first().isEmpty())
            db.store().save(Conversation("second", "Question", "{}", 2))
            db.store().clear()
            assertTrue(db.store().history().first().isEmpty())
            assertEquals(1, db.store().reports().first().size)
        } finally { db.close(); context.deleteDatabase(name) }
    }
}
