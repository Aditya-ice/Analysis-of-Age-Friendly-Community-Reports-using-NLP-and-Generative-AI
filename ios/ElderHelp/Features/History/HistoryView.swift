import SwiftData
import SwiftUI

struct HistoryView: View {
    @Environment(\.modelContext) private var modelContext
    @Query(sort: \StoredConversation.createdAt, order: .reverse) private var conversations: [StoredConversation]
    @State private var showClearConfirmation = false

    var body: some View {
        List {
            ForEach(conversations) { conversation in
                NavigationLink {
                    ConversationView(conversation: conversation)
                } label: {
                    VStack(alignment: .leading, spacing: 6) {
                        Text(conversation.question).font(.headline).lineLimit(2)
                        Text(conversation.createdAt, format: .dateTime.month().day().year().hour().minute())
                            .font(.footnote).foregroundStyle(.secondary)
                    }
                    .padding(.vertical, 6)
                }
            }
            .onDelete { offsets in
                for index in offsets { modelContext.delete(conversations[index]) }
            }
        }
        .overlay {
            if conversations.isEmpty {
                ContentUnavailableView(
                    "No saved questions",
                    systemImage: "clock.arrow.circlepath",
                    description: Text("Completed answers are saved only on this device.")
                )
            }
        }
        .navigationTitle("History")
        .toolbar {
            if !conversations.isEmpty {
                Button("Clear", role: .destructive) { showClearConfirmation = true }
            }
        }
        .confirmationDialog("Clear all history?", isPresented: $showClearConfirmation) {
            Button("Clear all history", role: .destructive) {
                for conversation in conversations { modelContext.delete(conversation) }
            }
        } message: {
            Text("This removes all saved questions and answers from this device.")
        }
    }
}

private struct ConversationView: View {
    let conversation: StoredConversation
    @State private var selectedCitation: Citation?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                Text("Question").font(.headline).accessibilityAddTraits(.isHeader)
                Text(conversation.question)
                Text("Answer").font(.headline).accessibilityAddTraits(.isHeader)
                Text((try? AttributedString(markdown: conversation.answerMarkdown)) ?? AttributedString(conversation.answerMarkdown))
                    .textSelection(.enabled)
                if !conversation.citations.isEmpty {
                    Text("Sources").font(.headline).accessibilityAddTraits(.isHeader)
                    ForEach(conversation.citations) { citation in
                        Button("[\(citation.id)] \(citation.reportTitle), page \(citation.pageNumber)") {
                            selectedCitation = citation
                        }
                        .frame(minHeight: 50, alignment: .leading)
                    }
                }
            }
            .padding()
        }
        .navigationTitle("Saved answer")
        .navigationBarTitleDisplayMode(.inline)
        .sheet(item: $selectedCitation) { CitationSheet(citation: $0) }
    }
}
