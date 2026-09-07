import SwiftUI

struct CitationSheet: View {
    let citation: Citation
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Text(citation.reportTitle).font(.title2.bold()).accessibilityAddTraits(.isHeader)
                    Text("Page \(citation.pageNumber) · \(citation.publisher)")
                        .foregroundStyle(.secondary)
                    Text(citation.excerpt).font(.body).textSelection(.enabled)
                    Link(destination: citation.sourceURL) {
                        Label("Open publisher's page", systemImage: "arrow.up.right.square")
                            .frame(minHeight: 50)
                    }
                    Text("This excerpt is provided for evidence. The publisher's page is the source of record.")
                        .font(.footnote).foregroundStyle(.secondary)
                }
                .padding()
            }
            .navigationTitle("Source \(citation.id)")
            .toolbar { Button("Done") { dismiss() } }
        }
    }
}
