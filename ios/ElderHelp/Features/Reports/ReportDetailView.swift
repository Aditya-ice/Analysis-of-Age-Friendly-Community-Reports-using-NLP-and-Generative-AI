import SwiftUI

struct ReportDetailView: View {
    let report: ReportSummary
    let apiClient: APIClient
    @State private var detail: ReportDetail?
    @State private var errorMessage: String?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                Text(report.title).font(.title.bold()).accessibilityAddTraits(.isHeader)
                Label(report.community, systemImage: "mappin.and.ellipse")
                Text(report.publisher).foregroundStyle(.secondary)
                if let date = report.publicationDate { Text("Published \(date)") }
                if let description = detail?.description { Text(description) }
                if let pageCount = detail?.pageCount { Text("\(pageCount) pages") }
                if let questions = detail?.suggestedQuestions, !questions.isEmpty {
                    Text("Questions to explore").font(.title2.bold()).accessibilityAddTraits(.isHeader)
                    ForEach(questions, id: \.self) { question in
                        NavigationLink(question) { AskView(apiClient: apiClient, reportID: report.id, question: question) }
                            .frame(minHeight: 44, alignment: .leading)
                    }
                }
                NavigationLink("Ask about this report") { AskView(apiClient: apiClient, reportID: report.id) }
                Link(destination: report.sourceURL) {
                    Label("Open publisher's page", systemImage: "arrow.up.right.square")
                        .frame(minHeight: 50)
                }
                if let errorMessage {
                    Text(errorMessage).foregroundStyle(.secondary)
                }
            }
            .padding()
        }
        .navigationTitle("Report")
        .navigationBarTitleDisplayMode(.inline)
        .task {
            do { detail = try await apiClient.report(id: report.id) }
            catch { errorMessage = "Detailed information is unavailable right now." }
        }
    }
}
