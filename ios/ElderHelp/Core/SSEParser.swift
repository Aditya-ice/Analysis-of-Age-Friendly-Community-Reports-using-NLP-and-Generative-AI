import Foundation

public struct RawSSEEvent: Equatable, Sendable {
    public let name: String
    public let data: String
}

public struct SSEParser: Sendable {
    private var eventName = "message"
    private var dataLines: [String] = []

    public init() {}

    public mutating func ingest(line: String) -> RawSSEEvent? {
        if line.isEmpty {
            guard !dataLines.isEmpty else {
                eventName = "message"
                return nil
            }
            let result = RawSSEEvent(name: eventName, data: dataLines.joined(separator: "\n"))
            eventName = "message"
            dataLines.removeAll(keepingCapacity: true)
            return result
        }
        if line.hasPrefix(":") {
            return nil
        }
        if line.hasPrefix("event:") {
            eventName = String(line.dropFirst(6)).trimmingCharacters(in: .whitespaces)
        } else if line.hasPrefix("data:") {
            dataLines.append(String(line.dropFirst(5)).trimmingCharacters(in: .whitespaces))
        }
        return nil
    }
}
