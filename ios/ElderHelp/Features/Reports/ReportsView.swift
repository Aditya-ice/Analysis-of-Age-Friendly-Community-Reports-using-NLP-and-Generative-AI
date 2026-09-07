import SwiftData
import SwiftUI

struct ReportsView: View {
    @State private var viewModel: ReportsViewModel
    @Environment(\.modelContext) private var modelContext
    @Query(sort: \CachedReport.title) private var cachedReports: [CachedReport]
    @State private var searchText = ""

    init(apiClient: APIClient) {
        _viewModel = State(initialValue: ReportsViewModel(apiClient: apiClient))
    }

    private var filtered: [ReportSummary] {
        guard !searchText.isEmpty else { return viewModel.reports }
        return viewModel.reports.filter {
            $0.title.localizedCaseInsensitiveContains(searchText)
                || $0.community.localizedCaseInsensitiveContains(searchText)
                || $0.publisher.localizedCaseInsensitiveContains(searchText)
        }
    }

    var body: some View {
        List(filtered) { report in
            NavigationLink {
                ReportDetailView(report: report, apiClient: viewModel.apiClient)
            } label: {
                VStack(alignment: .leading, spacing: 6) {
                    Text(report.title).font(.headline)
                    Text(report.community).font(.subheadline)
                    Text(report.publisher).font(.footnote).foregroundStyle(.secondary)
                }
                .padding(.vertical, 6)
            }
            .accessibilityHint("Shows report details and suggested questions")
        }
        .overlay {
            if viewModel.isLoading && viewModel.reports.isEmpty { ProgressView("Loading reports") }
            else if viewModel.reports.isEmpty { ContentUnavailableView("No reports", systemImage: "books.vertical") }
        }
        .searchable(text: $searchText, prompt: "Title, community, or publisher")
        .navigationTitle("Reports")
        .task {
            await viewModel.load(cached: cachedReports.compactMap(\.summary)) { fresh in
                for old in cachedReports { modelContext.delete(old) }
                for report in fresh { modelContext.insert(CachedReport(report)) }
            }
        }
        .safeAreaInset(edge: .bottom) {
            if let message = viewModel.errorMessage {
                Text(message).font(.footnote).padding(8).frame(maxWidth: .infinity)
                    .background(.yellow.opacity(0.18))
                    .accessibilityLabel("Notice: \(message)")
            }
        }
    }
}
