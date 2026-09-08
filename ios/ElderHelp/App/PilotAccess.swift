import Foundation
import Security
import Observation

// Device-only Keychain item, scoped to this API origin. Invite codes are never persisted.
enum PilotKeychain {
    static var account: String { AppConfiguration.apiBaseURL.absoluteString }
    static func read() -> String? {
        var result: CFTypeRef?
        let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: "com.adityaice.ElderHelp.pilot", kSecAttrAccount as String: account,
            kSecReturnData as String: true, kSecMatchLimit as String: kSecMatchLimitOne]
        guard SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess,
              let data = result as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }
    static func save(_ value: String?) throws {
        let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: "com.adityaice.ElderHelp.pilot", kSecAttrAccount as String: account]
        SecItemDelete(query as CFDictionary)
        guard let value else { return }
        var item = query
        item[kSecValueData as String] = Data(value.utf8)
        item[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
        guard SecItemAdd(item as CFDictionary, nil) == errSecSuccess else { throw APIError.invalidResponse }
    }
}

@MainActor @Observable final class PilotAccess {
    var invite = ""
    var connected = PilotKeychain.read() != nil
    var message: String?
    var busy = false
    let client: APIClient
    init(client: APIClient) { self.client = client }
    func connect() async {
        guard invite.count >= 12 else { message = "Enter the invite code supplied for this pilot."; return }
        busy = true; defer { busy = false }
        do {
            let session = try await client.connect(invite: invite)
            try PilotKeychain.save(session.token)
            invite = ""; connected = true
            let caps = try await client.capabilities()
            message = caps.generationAvailable ? "Connected. Answers appear after verification." : "Answers are temporarily unavailable. Use Search passages."
        } catch { message = connectionMessage(error) }
    }
    func disconnect() async {
        try? PilotKeychain.save(nil); await client.setToken(nil)
        connected = false; message = "Access token removed. Saved history is still available."
    }
}
