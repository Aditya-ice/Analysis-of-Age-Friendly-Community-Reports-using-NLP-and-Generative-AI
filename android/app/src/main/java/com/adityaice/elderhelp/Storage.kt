package com.adityaice.elderhelp

import androidx.room.*
import kotlinx.coroutines.flow.Flow

@Entity data class Conversation(@PrimaryKey val id: String, val question: String, val answer: String, val created: Long)
@Entity data class CachedReport(@PrimaryKey val id: String, val payload: String)
@Dao interface Store {
    @Query("SELECT * FROM Conversation ORDER BY created DESC") fun history(): Flow<List<Conversation>>
    @Query("SELECT * FROM CachedReport") fun reports(): Flow<List<CachedReport>>
    @Insert(onConflict = OnConflictStrategy.REPLACE) suspend fun save(conversation: Conversation)
    @Insert(onConflict = OnConflictStrategy.REPLACE) suspend fun saveReports(reports: List<CachedReport>)
    @Query("DELETE FROM CachedReport") suspend fun deleteReports()
    @Transaction suspend fun replaceReports(reports: List<CachedReport>) { deleteReports(); saveReports(reports) }
    @Query("DELETE FROM Conversation WHERE id = :id") suspend fun delete(id: String)
    @Query("DELETE FROM Conversation") suspend fun clear()
}
@Database(entities = [Conversation::class, CachedReport::class], version = 1, exportSchema = true)
abstract class LocalDatabase : RoomDatabase() { abstract fun store(): Store }
class Repository(val api: Api, val store: Store) {
    suspend fun refresh() = store.replaceReports(api.reports().items.map {
        CachedReport(it.id, codec.encodeToString(ReportSummary.serializer(), it))
    })
}
