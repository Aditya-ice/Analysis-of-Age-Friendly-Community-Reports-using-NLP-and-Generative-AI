import Foundation
import Observation

@MainActor
@Observable
final class ReportsViewModel {
    var reports: [ReportSummary] = []
    var isLoading = false
    var errorMessage: String?

    let apiClient: APIClient

    init(apiClient: APIClient) {
        self.apiClient = apiClient
    }

    func load(cached: [ReportSummary], onRefresh: ([ReportSummary]) -> Void) async {
        if reports.isEmpty { reports = cached }
        isLoading = true
        defer { isLoading = false }
        do {
            let fresh = try await apiClient.reports()
            reports = fresh
            errorMessage = nil
            onRefresh(fresh)
        } catch {
            errorMessage = cached.isEmpty ? error.localizedDescription : "Showing saved report information."
        }
    }
}
