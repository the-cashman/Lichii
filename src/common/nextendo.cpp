// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

// SPDX-FileCopyrightText: Copyright 2026 citron Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <atomic>
#include <cctype>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <nlohmann/json.hpp>
#include <openssl/rand.h>
#include <openssl/sha.h>

#include "common/fs/file.h"
#include "common/fs/fs.h"
#include "common/fs/path_util.h"
#include "common/httplib.h"
#include "common/logging.h"
#include "common/nextendo.h"
#include "common/settings.h"

#include "common/net/ca_store.h"

namespace Common::Nextendo {

namespace {

std::mutex account_mutex;
// Bumped by every sign-in and sign-out, so a renewal that started before one doesn't save over it.
u64 account_generation = 0;
std::optional<Account> cached_account;
bool account_loaded = false;
std::mutex refresh_mutex;
std::optional<std::chrono::steady_clock::time_point> last_renewal_failure;
std::mutex api_mutex;
std::string api_base{ApiBase};
std::deque<std::chrono::steady_clock::time_point> recent_requests;
std::atomic_bool sign_in_running{false};
SignOutReason sign_out_reason = SignOutReason::None;

std::filesystem::path AccountPath() {
    return FS::GetEdenPath(FS::EdenPath::ConfigDir) / "nextendo_account.json";
}

std::string PercentEncode(std::string_view text) {
    std::string out;
    for (const char c : text) {
        const auto u = static_cast<unsigned char>(c);
        if (std::isalnum(u) || c == '-' || c == '.' || c == '_' || c == '~') {
            out += c;
        } else {
            out += fmt::format("%{:02X}", u);
        }
    }
    return out;
}

std::optional<std::string> RandomToken() {
    std::array<u8, 32> raw{};
    if (RAND_bytes(raw.data(), static_cast<int>(raw.size())) != 1) {
        return std::nullopt;
    }
    return Base64UrlEncode(raw);
}

u64 ReadPid(const nlohmann::json& value) {
    if (value.is_number_unsigned()) {
        return value.get<u64>();
    }
    if (value.is_string()) {
        try {
            return std::stoull(value.get<std::string>());
        } catch (...) {
        }
    }
    return 0;
}

// Server fields may be null or another type; those read as unset instead of throwing.
std::string StringField(const nlohmann::json& object, const char* key, std::string fallback = {}) {
    const auto it = object.find(key);
    return it != object.end() && it->is_string() ? it->get<std::string>() : std::move(fallback);
}

void ReadIdentity(const nlohmann::json& object, u64& pid, std::string& username,
                  std::string& friend_code) {
    if (!object.is_object()) {
        return;
    }
    if (const auto it = object.find("pid"); it != object.end()) {
        pid = ReadPid(*it);
    }
    username = StringField(object, "username", username);
    friend_code = StringField(object, "friend_code", friend_code);
}

std::string ServerError(const std::string& body) {
    try {
        const auto json = nlohmann::json::parse(body);
        for (const char* key : {"error_description", "message", "error"}) {
            if (const auto it = json.find(key); it != json.end() && it->is_string()) {
                return it->get<std::string>();
            }
        }
    } catch (...) {
    }
    return {};
}

s64 Now() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

s64 IntField(const nlohmann::json& object, const char* key) {
    const auto it = object.find(key);
    return it != object.end() && it->is_number_integer() ? it->get<s64>() : 0;
}

// Every request to nextendo.network goes through here, so they all share the per-minute budget.
// A request over the budget isn't sent and fails as if the server were unreachable.
class ApiClient {
public:
    explicit ApiClient(time_t timeout_seconds) : client{Base()} {
        ConfigureClient(client, timeout_seconds);
    }

    httplib::Result Get(std::string_view path, const httplib::Headers& headers) {
        if (!TakeRequestSlot()) {
            return Skipped();
        }
        return client.Get(std::string{path}, headers);
    }

    httplib::Result Post(std::string_view path, const httplib::Headers& headers,
                         const httplib::Params& form) {
        if (!TakeRequestSlot()) {
            return Skipped();
        }
        return client.Post(std::string{path}, headers, form);
    }

    httplib::Result PostJson(std::string_view path, const httplib::Headers& headers,
                             const std::string& body) {
        if (!TakeRequestSlot()) {
            return Skipped();
        }
        return client.Post(std::string{path}, headers, body, "application/json");
    }

private:
    static std::string Base() {
        std::scoped_lock lock{api_mutex};
        return api_base;
    }

    static httplib::Result Skipped() {
        LOG_WARNING(Common, "Not sending a Nextendo request: {} were sent in the last minute",
                    RequestsPerMinute);
        return httplib::Result{nullptr, httplib::Error::Canceled};
    }

    httplib::Client client;
};

httplib::Headers BaseHeaders(std::string_view client_id) {
    return {{"User-Agent", "Lichii"}, {"X-Nextendo-Client-Id", std::string{client_id}}};
}

httplib::Headers BearerHeaders(std::string_view client_id, const std::string& token) {
    auto headers = BaseHeaders(client_id);
    headers.emplace("Authorization", "Bearer " + token);
    return headers;
}

std::optional<TokenResponse> ReadReply(const httplib::Result& result, std::string_view what,
                                       std::string& error) {
    if (!result) {
        error = "Could not reach nextendo.network.";
        LOG_ERROR(Common, "Nextendo {} failed: {}", what, httplib::to_string(result.error()));
        return std::nullopt;
    }
    if (result->status != 200) {
        const std::string reason = ServerError(result->body);
        error = fmt::format("nextendo.network refused the {} (HTTP {}){}{}", what, result->status,
                            reason.empty() ? "" : ": ", reason);
        LOG_ERROR(Common, "{}", error);
        return std::nullopt;
    }
    auto token = ParseTokenResponse(result->body);
    if (!token) {
        error = fmt::format("nextendo.network sent an unreadable {} response.", what);
        return std::nullopt;
    }
    LOG_INFO(Common, "Nextendo {} response fields: {}", what, fmt::join(token->fields, ", "));
    return token;
}

void ApplyIdentity(Account& account, const TokenResponse& token) {
    if (token.pid != 0) {
        account.pid = token.pid;
    }
    if (!token.username.empty()) {
        account.username = token.username;
    }
    if (!token.friend_code.empty()) {
        account.friend_code = token.friend_code;
    }
}

void ApplyTokens(Account& account, const TokenResponse& token) {
    account.access_token = token.access_token;
    account.access_expires = token.expires_in > 0 ? Now() + token.expires_in : 0;
    if (!token.refresh_token.empty()) {
        account.refresh_token = token.refresh_token;
    }
    if (!token.nex_token.empty()) {
        account.nex_token = token.nex_token;
        account.nex_expires = 0;
    }
    ApplyIdentity(account, token);
}

enum class GameTokenReply {
    Ok,
    // Unreachable, rate limited or an unreadable answer: worth retrying later.
    Failed,
    // HTTP 401: the access token expired or the player removed Lichii's access.
    Unauthorized,
    // HTTP 403: the sign-in didn't grant game.matchmaking.
    MissingScope,
};

// Third-party clients trade the access token for the day-long nx2 game token that NEX servers
// accept (wiki.nextendo.network/developers, "Game login").
GameTokenReply FetchGameToken(ApiClient& api, std::string_view client_id, Account& account,
                              std::string& error) {
    const auto result = api.Get(NexTokenPath, BearerHeaders(client_id, account.access_token));
    if (result && (result->status == 401 || result->status == 403)) {
        LOG_WARNING(Common, "nextendo.network refused the game token request (HTTP {})",
                    result->status);
        if (result->status == 401) {
            error = "nextendo.network no longer accepts this sign-in.";
            return GameTokenReply::Unauthorized;
        }
        error = "The sign-in didn't allow online play. Sign in again and allow it.";
        return GameTokenReply::MissingScope;
    }
    const auto token = ReadReply(result, "game token", error);
    if (!token) {
        return GameTokenReply::Failed;
    }
    if (token->nex_token.empty()) {
        error = "nextendo.network sent no game token.";
        return GameTokenReply::Failed;
    }
    account.nex_token = token->nex_token;
    account.nex_expires = Now() + (token->expires_in > 0 ? token->expires_in : 24 * 60 * 60);
    ApplyIdentity(account, *token);
    return GameTokenReply::Ok;
}

void ExpireGameToken(const std::string& nex_token);

// A 401 means the game token was ended before it expired, so the next renewal replaces it.
std::optional<OnlineStatus> FetchOnlineStatus(ApiClient& api, std::string_view client_id,
                                              const std::string& nex_token) {
    const auto result = api.Get(OnlineStatusPath, BearerHeaders(client_id, nex_token));
    if (!result || result->status != 200) {
        LOG_WARNING(Common, "Nextendo online status request failed (HTTP {})",
                    result ? result->status : 0);
        if (result && result->status == 401) {
            ExpireGameToken(nex_token);
        }
        return std::nullopt;
    }
    return ParseOnlineStatus(result->body);
}

// /api/me: the identity when the token replies lacked it, and the BAAS user id.
void FetchUserInfo(ApiClient& api, std::string_view client_id, Account& account) {
    const auto info = api.Get(UserInfoPath, BearerHeaders(client_id, account.access_token));
    if (!info || info->status != 200) {
        LOG_WARNING(Common, "Nextendo user info request failed (HTTP {})", info ? info->status : 0);
        return;
    }
    try {
        const auto json = nlohmann::json::parse(info->body);
        const bool needs_identity = account.pid == 0;
        for (const auto& object : {json, json.value("account", nlohmann::json{})}) {
            if (needs_identity) {
                ReadIdentity(object, account.pid, account.username, account.friend_code);
            }
            if (const u64 nsa = object.is_object() ? ParseBaasNsa(StringField(object, "nsa")) : 0;
                nsa != 0) {
                account.baas_nsa = nsa;
            }
        }
    } catch (...) {
    }
}

std::optional<Account> LoadAccountLocked() {
    if (account_loaded) {
        return cached_account;
    }
    account_loaded = true;
    const std::string text = FS::ReadStringFromFile(AccountPath(), FS::FileType::TextFile);
    if (text.empty()) {
        return cached_account;
    }
    try {
        const auto json = nlohmann::json::parse(text);
        Account account;
        ReadIdentity(json, account.pid, account.username, account.friend_code);
        account.nex_token = StringField(json, "nex_token");
        account.access_token = StringField(json, "access_token");
        account.refresh_token = StringField(json, "refresh_token");
        account.access_expires = IntField(json, "access_expires");
        account.nex_expires = IntField(json, "nex_expires");
        account.baas_nsa = ParseBaasNsa(StringField(json, "nsa"));
        cached_account = std::move(account);
    } catch (const std::exception& e) {
        LOG_WARNING(Common, "Ignoring an unreadable Nextendo account file: {}", e.what());
    }
    return cached_account;
}

// The file holds the refresh token, so only its owner may read it. It is written beside the old
// one and renamed over it, so a crash mid-write keeps the previous session. The session stays in
// memory when the write fails, so a rotated refresh token isn't lost until Lichii exits.
bool WriteAccountLocked(const Account& account) {
    cached_account = account;
    account_loaded = true;
    const std::string nsa = account.baas_nsa != 0 ? fmt::format("{:016x}", account.baas_nsa) : "";
    const std::string text = nlohmann::json{
        {"pid", account.pid},
        {"username", account.username},
        {"friend_code", account.friend_code},
        {"nex_token", account.nex_token},
        {"access_token", account.access_token},
        {"refresh_token", account.refresh_token},
        {"access_expires", account.access_expires},
        {"nex_expires", account.nex_expires},
        {"nsa", nsa},
    }.dump(2);
    const auto path = AccountPath();
    auto temp = path;
    temp += ".tmp";
    void(FS::CreateParentDirs(path));
    void(FS::WriteStringToFile(temp, FS::FileType::BinaryFile, ""));
    std::error_code ec;
    std::filesystem::permissions(temp,
                                 std::filesystem::perms::owner_read |
                                     std::filesystem::perms::owner_write,
                                 std::filesystem::perm_options::replace, ec);
    if (FS::WriteStringToFile(temp, FS::FileType::BinaryFile, text) != text.size()) {
        LOG_ERROR(Common, "Could not write the Nextendo account file");
        void(FS::RemoveFile(temp));
        return false;
    }
    std::filesystem::rename(temp, path, ec);
    if (ec) {
        LOG_ERROR(Common, "Could not replace the Nextendo account file: {}", ec.message());
        void(FS::RemoveFile(temp));
        return false;
    }
    return true;
}

void ClearAccountLocked() {
    ++account_generation;
    void(FS::RemoveFile(AccountPath()));
    cached_account.reset();
    account_loaded = true;
    sign_out_reason = SignOutReason::None;
}

bool SaveRenewedAccount(const Account& account, u64 generation) {
    std::scoped_lock lock{account_mutex};
    return generation == account_generation && WriteAccountLocked(account);
}

bool ClearRenewedAccount(u64 generation, SignOutReason reason) {
    std::scoped_lock lock{account_mutex};
    if (generation != account_generation) {
        return false;
    }
    ClearAccountLocked();
    sign_out_reason = reason;
    return true;
}

void ExpireGameToken(const std::string& nex_token) {
    std::scoped_lock lock{account_mutex};
    auto account = LoadAccountLocked();
    if (account && account->nex_token == nex_token && account->nex_expires != 0) {
        account->nex_expires = 0;
        void(WriteAccountLocked(*account));
    }
}

} // namespace

void ConfigureClient(httplib::Client& client, time_t timeout_seconds) {
    client.set_connection_timeout(timeout_seconds);
    client.set_read_timeout(timeout_seconds);
    client.set_write_timeout(timeout_seconds);
    client.set_follow_location(true);
    Common::Net::ConfigureCaStore(client);
}

bool TakeRequestSlot() {
    const auto now = std::chrono::steady_clock::now();
    std::scoped_lock lock{api_mutex};
    while (!recent_requests.empty() && now - recent_requests.front() >= std::chrono::minutes{1}) {
        recent_requests.pop_front();
    }
    if (recent_requests.size() >= RequestsPerMinute) {
        return false;
    }
    recent_requests.push_back(now);
    return true;
}

bool OnlinePlayActive() {
    return Settings::values.nextendo_enabled.GetValue();
}

void SetApiBaseForTests(std::string base) {
    {
        std::scoped_lock lock{api_mutex};
        api_base = std::move(base);
        recent_requests.clear();
    }
    std::scoped_lock lock{refresh_mutex};
    last_renewal_failure.reset();
}

std::optional<Account> LoadAccount() {
    std::scoped_lock lock{account_mutex};
    return LoadAccountLocked();
}

void SaveAccount(const Account& account) {
    std::scoped_lock lock{account_mutex};
    ++account_generation;
    sign_out_reason = SignOutReason::None;
    void(WriteAccountLocked(account));
}

void ClearAccount() {
    std::scoped_lock lock{account_mutex};
    ClearAccountLocked();
}

SignOutReason LastSignOutReason() {
    std::scoped_lock lock{account_mutex};
    return sign_out_reason;
}

u64 NetworkServiceAccountId(u64 program_id, const Account& account) {
    return program_id == AnimalCrossingNewHorizons && account.baas_nsa != 0 ? account.baas_nsa
                                                                            : account.pid;
}

u64 ParseBaasNsa(std::string_view text) {
    u64 nsa = 0;
    const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), nsa, 16);
    return text.size() == 16 && ec == std::errc{} && end == text.data() + text.size() ? nsa : 0;
}

bool RefreshTokenRejected(std::string_view body) {
    try {
        const auto json = nlohmann::json::parse(body);
        return json.is_object() && StringField(json, "error") == "invalid_grant";
    } catch (...) {
        return false;
    }
}

std::string Base64UrlEncode(std::span<const u8> data) {
    static constexpr std::string_view alphabet =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    std::string out;
    out.reserve((data.size() + 2) / 3 * 4);
    for (std::size_t i = 0; i < data.size(); i += 3) {
        const std::size_t remaining = data.size() - i;
        const u32 triple = (u32{data[i]} << 16) | (remaining > 1 ? u32{data[i + 1]} << 8 : 0) |
                           (remaining > 2 ? u32{data[i + 2]} : 0);
        out += alphabet[(triple >> 18) & 0x3F];
        out += alphabet[(triple >> 12) & 0x3F];
        if (remaining > 1) {
            out += alphabet[(triple >> 6) & 0x3F];
        }
        if (remaining > 2) {
            out += alphabet[triple & 0x3F];
        }
    }
    return out;
}

std::string PkceChallenge(std::string_view verifier) {
    std::array<u8, SHA256_DIGEST_LENGTH> digest{};
    SHA256(reinterpret_cast<const unsigned char*>(verifier.data()), verifier.size(),
           digest.data());
    return Base64UrlEncode(digest);
}

std::optional<Pkce> MakePkce() {
    auto verifier = RandomToken();
    if (!verifier) {
        return std::nullopt;
    }
    std::string challenge = PkceChallenge(*verifier);
    return Pkce{.verifier = std::move(*verifier), .challenge = std::move(challenge)};
}

std::string RedirectUri(u16 port) {
    return fmt::format("http://127.0.0.1:{}/callback", port);
}

std::string BuildAuthorizeUrl(std::string_view client_id, std::string_view state,
                              std::string_view challenge, u16 port) {
    return fmt::format("{}{}?response_type=code&client_id={}&redirect_uri={}&scope={}&state={}"
                       "&code_challenge={}&code_challenge_method=S256",
                       ApiBase, AuthorizePath, PercentEncode(client_id),
                       PercentEncode(RedirectUri(port)), PercentEncode(Scopes),
                       PercentEncode(state), PercentEncode(challenge));
}

Callback ParseCallback(const std::multimap<std::string, std::string>& params,
                       std::string_view expected_state) {
    const auto get = [&params](const char* key) -> std::string {
        const auto it = params.find(key);
        return it == params.end() ? std::string{} : it->second;
    };
    if (expected_state.empty() || get("state") != expected_state) {
        return {.error = "The sign-in reply did not match this request."};
    }
    if (const std::string error = get("error"); !error.empty()) {
        const std::string description = get("error_description");
        return {.matched = true,
                .error = description.empty() ? error : fmt::format("{}: {}", error, description)};
    }
    std::string code = get("code");
    if (code.empty()) {
        return {.matched = true, .error = "The browser returned no authorization code."};
    }
    return {.matched = true, .code = std::move(code)};
}

std::optional<TokenResponse> ParseTokenResponse(std::string_view body) {
    nlohmann::json json;
    try {
        json = nlohmann::json::parse(body);
    } catch (...) {
        return std::nullopt;
    }
    if (!json.is_object()) {
        return std::nullopt;
    }
    TokenResponse out;
    for (const auto& [key, value] : json.items()) {
        out.fields.push_back(key);
    }
    out.nex_token = StringField(json, "nex_token");
    out.access_token = StringField(json, "access_token");
    out.refresh_token = StringField(json, "refresh_token");
    out.expires_in = IntField(json, "expires_in");
    ReadIdentity(json, out.pid, out.username, out.friend_code);
    if (const auto it = json.find("account"); it != json.end()) {
        ReadIdentity(*it, out.pid, out.username, out.friend_code);
    }
    return out;
}

std::optional<OnlineStatus> ParseOnlineStatus(std::string_view body) {
    try {
        const auto json = nlohmann::json::parse(body);
        const auto allow = json.find("allow");
        if (!json.is_object() || allow == json.end() || !allow->is_boolean()) {
            return std::nullopt;
        }
        return OnlineStatus{.allow = allow->get<bool>(), .reason = StringField(json, "reason")};
    } catch (...) {
        return std::nullopt;
    }
}

SignInResult SignIn(std::string_view client_id,
                    const std::function<void(const std::string&)>& open_url,
                    std::stop_token stop) {
    SignInResult out;
    if (client_id.empty()) {
        out.error = "No Nextendo client id is set.";
        return out;
    }
    // A second sign-in would listen on the same port and take the first one's callback.
    if (sign_in_running.exchange(true)) {
        out.error = "A sign-in is already running.";
        return out;
    }
    struct Running {
        ~Running() {
            sign_in_running = false;
        }
    } running;

    const auto pkce = MakePkce();
    const auto state = RandomToken();
    if (!pkce || !state) {
        out.error = "Lichii could not generate a random sign-in key.";
        return out;
    }

    std::mutex mutex;
    std::condition_variable_any cv;
    std::optional<Callback> callback;

    const auto make_server = [&] {
        auto server = std::make_unique<httplib::Server>();
        server->set_socket_options([](socket_t sock) {
            int yes = 1;
#ifdef _WIN32
            void(setsockopt(sock, SOL_SOCKET, SO_EXCLUSIVEADDRUSE,
                            reinterpret_cast<const char*>(&yes), sizeof(yes)));
#else
            void(setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes)));
#endif
        });
        server->Get("/callback", [&](const httplib::Request& request,
                                     httplib::Response& response) {
            const std::multimap<std::string, std::string> params(request.params.begin(),
                                                                 request.params.end());
            Callback parsed = ParseCallback(params, *state);
            if (!parsed.matched) {
                response.status = 400;
                response.set_content(parsed.error, "text/plain; charset=utf-8");
                return;
            }
            response.set_content(parsed.error.empty()
                                     ? "Signed in to Nextendo. You can close this tab and go back "
                                       "to Lichii."
                                     : "Nextendo sign-in failed: " + parsed.error,
                                 "text/plain; charset=utf-8");
            {
                std::scoped_lock lock{mutex};
                if (!callback) {
                    callback = std::move(parsed);
                }
            }
            cv.notify_all();
        });
        return server;
    };
    auto server = make_server();
    int port = CallbackPort;
    if (!server->bind_to_port("127.0.0.1", CallbackPort)) {
        // A failed bind leaves the server unusable. nextendo.network accepts any loopback port
        // (RFC 8252), so a new one takes a free port.
        server = make_server();
        port = server->bind_to_any_port("127.0.0.1");
        if (port <= 0) {
            out.error = "Lichii could not listen for the sign-in reply on 127.0.0.1.";
            return out;
        }
        LOG_INFO(Common, "127.0.0.1:{} is taken; the Nextendo sign-in listens on port {}",
                 CallbackPort, port);
    }
    const u16 callback_port = static_cast<u16>(port);
    std::thread listener{[&server] { void(server->listen_after_bind()); }};
    // stop() is a no-op until the listener runs, so a cancel before then would never end it.
    server->wait_until_ready();

    open_url(BuildAuthorizeUrl(client_id, *state, pkce->challenge, callback_port));
    {
        std::unique_lock lock{mutex};
        cv.wait_for(lock, stop, std::chrono::minutes{5},
                    [&callback] { return callback.has_value(); });
    }
    server->stop();
    listener.join();

    if (!callback) {
        out.error = stop.stop_requested() ? "Sign-in was cancelled." : "Sign-in timed out.";
        return out;
    }
    if (!callback->error.empty()) {
        out.error = callback->error;
        return out;
    }

    ApiClient api{15};
    const httplib::Params form{
        {"grant_type", "authorization_code"}, {"code", callback->code},
        {"client_id", std::string{client_id}}, {"redirect_uri", RedirectUri(callback_port)},
        {"code_verifier", pkce->verifier},
    };
    const auto token = ReadReply(api.Post(TokenPath, BaseHeaders(client_id), form), "sign-in",
                                 out.error);
    if (!token) {
        return out;
    }
    out.token_fields = token->fields;
    if (token->access_token.empty() && token->nex_token.empty()) {
        out.error = "nextendo.network returned no token.";
        return out;
    }

    Account& account = out.account;
    ApplyTokens(account, *token);
    if (account.nex_token.empty() &&
        FetchGameToken(api, client_id, account, out.game_token_error) != GameTokenReply::Ok) {
        LOG_WARNING(Common, "Signed in to Nextendo without a game token: {}",
                    out.game_token_error);
    }
    if (!account.nex_token.empty()) {
        out.online_status = FetchOnlineStatus(api, client_id, account.nex_token);
    }

    if (!account.access_token.empty()) {
        FetchUserInfo(api, client_id, account);
    }

    SaveAccount(account);
    out.ok = true;
    return out;
}

std::optional<Account> EnsureGameToken(std::string_view client_id) {
    std::scoped_lock lock{refresh_mutex};
    std::optional<Account> account;
    u64 generation;
    {
        std::scoped_lock account_lock{account_mutex};
        account = LoadAccountLocked();
        generation = account_generation;
    }
    const s64 now = Now();
    if (!account || client_id.empty() ||
        (!account->nex_token.empty() && account->nex_expires > now + RenewAheadSeconds) ||
        (account->access_token.empty() && account->refresh_token.empty())) {
        return account;
    }
    // A game asks for its id token on the IPC thread, so an unreachable server is retried at most
    // every five minutes and with short timeouts.
    const auto steady_now = std::chrono::steady_clock::now();
    if (last_renewal_failure && steady_now - *last_renewal_failure < std::chrono::minutes{5}) {
        return account;
    }

    ApiClient api{5};
    std::string error;
    const auto refresh = [&] {
        if (account->refresh_token.empty()) {
            LOG_WARNING(Common, "The Nextendo session expired; sign in again");
            ClearRenewedAccount(generation, SignOutReason::SessionExpired);
            return false;
        }
        const httplib::Params form{
            {"grant_type", "refresh_token"},
            {"refresh_token", account->refresh_token},
            {"client_id", std::string{client_id}},
        };
        const auto result = api.Post(TokenPath, BaseHeaders(client_id), form);
        const auto token = ReadReply(result, "token refresh", error);
        if (!token) {
            if (result && RefreshTokenRejected(result->body)) {
                LOG_WARNING(Common, "nextendo.network refused the refresh token; signing out");
                ClearRenewedAccount(generation, SignOutReason::SessionExpired);
            } else {
                last_renewal_failure = steady_now;
            }
            return false;
        }
        ApplyTokens(*account, *token);
        // The server retires the old refresh token a minute after handing out a new one.
        if (!SaveRenewedAccount(*account, generation) || account->access_token.empty()) {
            last_renewal_failure = steady_now;
            return false;
        }
        return true;
    };

    bool refreshed = false;
    if (account->access_token.empty() || account->access_expires <= now + 60) {
        if (!refresh()) {
            return LoadAccount();
        }
        refreshed = true;
    }
    auto reply = FetchGameToken(api, client_id, *account, error);
    if (reply == GameTokenReply::Unauthorized && !refreshed) {
        // The access token was ended early; a new one from the refresh token may still work.
        if (!refresh()) {
            return LoadAccount();
        }
        reply = FetchGameToken(api, client_id, *account, error);
    }
    switch (reply) {
    case GameTokenReply::Ok:
        if (account->baas_nsa == 0) {
            FetchUserInfo(api, client_id, *account);
        }
        void(SaveRenewedAccount(*account, generation));
        break;
    case GameTokenReply::Failed:
        last_renewal_failure = steady_now;
        LOG_WARNING(Common, "Could not renew the Nextendo game token: {}", error);
        break;
    case GameTokenReply::Unauthorized:
        LOG_WARNING(Common, "nextendo.network refused a new access token; signing out");
        ClearRenewedAccount(generation, SignOutReason::SessionRevoked);
        break;
    case GameTokenReply::MissingScope:
        LOG_WARNING(Common, "The Nextendo sign-in lacks game.matchmaking; signing out");
        ClearRenewedAccount(generation, SignOutReason::MissingScope);
        break;
    }
    // What is stored now, which a sign-in or sign-out during the requests may have replaced.
    return LoadAccount();
}

std::optional<OnlineStatus> CheckOnlineStatus(std::string_view client_id) {
    const auto account = EnsureGameToken(client_id);
    if (!account || account->nex_token.empty()) {
        return std::nullopt;
    }
    ApiClient api{5};
    return FetchOnlineStatus(api, client_id, account->nex_token);
}

std::optional<std::string> FetchGameCatalog(std::string_view client_id) {
    ApiClient api{5};
    const auto result = api.Get(GameCatalogPath, BaseHeaders(client_id));
    if (!result || result->status != 200) {
        LOG_WARNING(Common, "Nextendo game catalogue request failed (HTTP {})",
                    result ? result->status : 0);
        return std::nullopt;
    }
    return result->body;
}

void RevokeToken(std::string_view client_id, const std::string& refresh_token) {
    ApiClient api{5};
    const httplib::Params form{{"token", refresh_token}, {"client_id", std::string{client_id}}};
    if (const auto result = api.Post(RevokePath, BaseHeaders(client_id), form);
        !result || result->status != 200) {
        LOG_WARNING(Common, "Could not revoke the Nextendo sign-in (HTTP {})",
                    result ? result->status : 0);
    }
}

SignOutReason KeepSessionAlive(std::string_view client_id) {
    u64 generation;
    {
        std::scoped_lock lock{account_mutex};
        if (!LoadAccountLocked()) {
            return SignOutReason::None;
        }
        generation = account_generation;
    }
    void(EnsureGameToken(client_id));
    std::optional<Account> account;
    {
        std::scoped_lock lock{account_mutex};
        account = LoadAccountLocked();
        if (generation != account_generation) {
            return account ? SignOutReason::None : sign_out_reason;
        }
    }
    if (!account || account->nex_token.empty()) {
        return SignOutReason::None;
    }
    ApiClient api{5};
    const auto headers = BearerHeaders(client_id, account->nex_token);
    const auto touched = api.PostJson(NexSessionPath, headers, "{}");
    if (!touched || touched->status != 200) {
        LOG_WARNING(Common, "Nextendo session heartbeat failed (HTTP {})",
                    touched ? touched->status : 0);
        if (touched && touched->status == 401) {
            ExpireGameToken(account->nex_token);
            return SignOutReason::None;
        }
    }
    const auto status = api.Get(SessionStatusPath, headers);
    if (!status || status->status != 200) {
        if (status && status->status == 401) {
            ExpireGameToken(account->nex_token);
        }
        return SignOutReason::None;
    }
    try {
        const auto json = nlohmann::json::parse(status->body);
        const auto revoked = json.find("revoked");
        if (!json.is_object() || revoked == json.end() || !revoked->is_boolean() ||
            !revoked->get<bool>()) {
            return SignOutReason::None;
        }
    } catch (...) {
        return SignOutReason::None;
    }
    LOG_WARNING(Common, "nextendo.network ended this device's session; signing out");
    return ClearRenewedAccount(generation, SignOutReason::SessionRevoked)
               ? SignOutReason::SessionRevoked
               : SignOutReason::None;
}

} // namespace Common::Nextendo
