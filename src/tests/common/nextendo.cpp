// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include <array>
#include <chrono>
#include <ctime>
#include <deque>
#include <filesystem>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <thread>

#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>
#include <openssl/evp.h>

#include "common/fs/path_util.h"
#include "common/httplib.h"
#include "common/nextendo.h"
#include "common/nextendo_game_catalog.h"
#include "common/settings.h"

namespace Nextendo = Common::Nextendo;

TEST_CASE("Nextendo: PKCE S256 matches RFC 7636 appendix B", "[common]") {
    REQUIRE(Nextendo::PkceChallenge("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk") ==
            "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM");

    const auto pkce = Nextendo::MakePkce();
    REQUIRE(pkce.has_value());
    REQUIRE(pkce->verifier.size() == 43);
    REQUIRE(pkce->challenge == Nextendo::PkceChallenge(pkce->verifier));
}

TEST_CASE("Nextendo: base64url has no padding", "[common]") {
    const auto encode = [](std::string_view text) {
        return Nextendo::Base64UrlEncode(
            std::span{reinterpret_cast<const u8*>(text.data()), text.size()});
    };
    REQUIRE(encode("f") == "Zg");
    REQUIRE(encode("fo") == "Zm8");
    REQUIRE(encode("foo") == "Zm9v");
    REQUIRE(encode("\xfb\xff") == "-_8");
}

TEST_CASE("Nextendo: the authorize URL carries the loopback redirect and PKCE", "[common]") {
    const std::string url = Nextendo::BuildAuthorizeUrl("nxc_test", "st", "ch", 47823);
    REQUIRE(url.starts_with("https://nextendo.network/oauth/authorize?response_type=code"));
    REQUIRE(url.find("&client_id=nxc_test") != std::string::npos);
    REQUIRE(url.find("&redirect_uri=http%3A%2F%2F127.0.0.1%3A47823%2Fcallback") !=
            std::string::npos);
    REQUIRE(url.find("&scope=identity%20game.matchmaking") != std::string::npos);
    REQUIRE(url.find("&state=st&code_challenge=ch&code_challenge_method=S256") !=
            std::string::npos);
}

TEST_CASE("Nextendo: callback parsing", "[common]") {
    using Params = std::multimap<std::string, std::string>;

    const auto ok = Nextendo::ParseCallback(Params{{"code", "abc"}, {"state", "s1"}}, "s1");
    REQUIRE(ok.matched);
    REQUIRE(ok.error.empty());
    REQUIRE(ok.code == "abc");

    const auto wrong_state =
        Nextendo::ParseCallback(Params{{"code", "abc"}, {"state", "other"}}, "s1");
    REQUIRE_FALSE(wrong_state.matched);
    REQUIRE(wrong_state.code.empty());

    // A stray local request with an error but not this sign-in's state must not end it.
    const auto stray = Nextendo::ParseCallback(Params{{"error", "access_denied"}}, "s1");
    REQUIRE_FALSE(stray.matched);

    const auto denied = Nextendo::ParseCallback(
        Params{{"error", "access_denied"}, {"error_description", "no"}, {"state", "s1"}}, "s1");
    REQUIRE(denied.matched);
    REQUIRE(denied.error == "access_denied: no");

    const auto no_code = Nextendo::ParseCallback(Params{{"state", "s1"}}, "s1");
    REQUIRE(no_code.code.empty());
    REQUIRE_FALSE(no_code.error.empty());
}

TEST_CASE("Nextendo: token responses", "[common]") {
    const auto emulator = Nextendo::ParseTokenResponse(
        R"({"token_type":"Bearer","nex_token":"nx2.a.b",)"
        R"("account":{"pid":1800000001,"username":"P","friend_code":"SW-1"}})");
    REQUIRE(emulator.has_value());
    REQUIRE(emulator->nex_token == "nx2.a.b");
    REQUIRE(emulator->pid == 1800000001);
    REQUIRE(emulator->username == "P");
    REQUIRE(emulator->friend_code == "SW-1");

    const auto third_party = Nextendo::ParseTokenResponse(
        R"({"access_token":"at","refresh_token":"rt","expires_in":3600})");
    REQUIRE(third_party.has_value());
    REQUIRE(third_party->nex_token.empty());
    REQUIRE(third_party->access_token == "at");
    REQUIRE(third_party->refresh_token == "rt");
    REQUIRE(third_party->fields.size() == 3);

    const auto wrong_types = Nextendo::ParseTokenResponse(
        R"({"access_token":"at","refresh_token":null,"nex_token":7,)"
        R"("account":{"pid":"42","username":null,"friend_code":[]}})");
    REQUIRE(wrong_types.has_value());
    REQUIRE(wrong_types->access_token == "at");
    REQUIRE(wrong_types->refresh_token.empty());
    REQUIRE(wrong_types->nex_token.empty());
    REQUIRE(wrong_types->pid == 42);
    REQUIRE(wrong_types->username.empty());
    REQUIRE(wrong_types->friend_code.empty());

    const auto game = Nextendo::ParseTokenResponse(
        R"({"nex_token":"nx2.g.s","pid":1800000002,"friend_code":"SW-2","username":"Q",)"
        R"("expires_in":86400})");
    REQUIRE(game.has_value());
    REQUIRE(game->nex_token == "nx2.g.s");
    REQUIRE(game->pid == 1800000002);
    REQUIRE(game->friend_code == "SW-2");
    REQUIRE(game->username == "Q");
    REQUIRE(game->expires_in == 86400);

    REQUIRE_FALSE(Nextendo::ParseTokenResponse("not json").has_value());
}

TEST_CASE("Nextendo: only invalid_grant ends the session", "[common]") {
    REQUIRE(Nextendo::RefreshTokenRejected(R"({"error":"invalid_grant"})"));
    REQUIRE(Nextendo::RefreshTokenRejected(
        R"({"error":"invalid_grant","error_description":"refresh token expired"})"));
    REQUIRE_FALSE(Nextendo::RefreshTokenRejected(R"({"error":"invalid_client"})"));
    REQUIRE_FALSE(Nextendo::RefreshTokenRejected(R"({"error":null})"));
    REQUIRE_FALSE(Nextendo::RefreshTokenRejected("<html>Bad Gateway</html>"));
}

TEST_CASE("Nextendo: online status reply", "[common]") {
    const auto refused = Nextendo::ParseOnlineStatus(R"({"allow":false,"reason":"unverified"})");
    REQUIRE(refused.has_value());
    REQUIRE_FALSE(refused->allow);
    REQUIRE(refused->reason == "unverified");

    const auto allowed = Nextendo::ParseOnlineStatus(R"({"allow":true,"reason":""})");
    REQUIRE(allowed.has_value());
    REQUIRE(allowed->allow);

    REQUIRE_FALSE(Nextendo::ParseOnlineStatus(R"({"error":"Session invalide"})").has_value());
    REQUIRE_FALSE(Nextendo::ParseOnlineStatus("not json").has_value());
}

namespace {

using namespace std::chrono_literals;

s64 Now() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

Nextendo::Account SignedIn(s64 game_token_left, s64 access_token_left) {
    return {.pid = 1,
            .username = "tester",
            .nex_token = "nx2.old",
            .access_token = "access.old",
            .refresh_token = "refresh.old",
            .access_expires = Now() + access_token_left,
            .nex_expires = Now() + game_token_left};
}

// Answers the API on 127.0.0.1 in place of nextendo.network. Each path replies with its queued
// answers in order and repeats the last one.
class FakeNextendo {
public:
    FakeNextendo() {
        directory = std::filesystem::temp_directory_path() /
                    fmt::format("lichii-nextendo-test-{}", std::random_device{}());
        std::filesystem::create_directories(directory);
        previous_config = Common::FS::GetEdenPath(Common::FS::EdenPath::ConfigDir);
        previous_cache = Common::FS::GetEdenPath(Common::FS::EdenPath::CacheDir);
        Common::FS::SetEdenPath(Common::FS::EdenPath::ConfigDir, directory);
        Common::FS::SetEdenPath(Common::FS::EdenPath::CacheDir, directory);

        const auto handle = [this](const httplib::Request& request, httplib::Response& response) {
            std::scoped_lock lock{mutex};
            ++hits[request.path];
            authorization[request.path] = request.get_header_value("Authorization");
            forms[request.path] = request.params;
            auto& queue = replies[request.path];
            const Canned reply = queue.empty() ? Canned{404, "{}"} : queue.front();
            if (queue.size() > 1) {
                queue.pop_front();
            }
            response.status = reply.status;
            response.set_content(reply.body, "application/json");
        };
        server.Get(".*", handle);
        server.Post(".*", handle);
        port = server.bind_to_any_port("127.0.0.1");
        REQUIRE(port > 0);
        thread = std::thread{[this] { void(server.listen_after_bind()); }};
        server.wait_until_ready();
        Nextendo::SetApiBaseForTests(fmt::format("http://127.0.0.1:{}", port));
        Nextendo::ClearAccount();
    }

    ~FakeNextendo() {
        server.stop();
        thread.join();
        Nextendo::ClearAccount();
        Nextendo::SetApiBaseForTests(std::string{Nextendo::ApiBase});
        Common::FS::SetEdenPath(Common::FS::EdenPath::ConfigDir, previous_config);
        Common::FS::SetEdenPath(Common::FS::EdenPath::CacheDir, previous_cache);
        std::error_code ec;
        std::filesystem::remove_all(directory, ec);
    }

    void Answer(const std::string& path, int status, std::string body) {
        std::scoped_lock lock{mutex};
        replies[path].push_back({status, std::move(body)});
    }

    int Hits(const std::string& path) {
        std::scoped_lock lock{mutex};
        return hits[path];
    }

    int TotalHits() {
        std::scoped_lock lock{mutex};
        int total = 0;
        for (const auto& [path, count] : hits) {
            total += count;
        }
        return total;
    }

    std::string Authorization(const std::string& path) {
        std::scoped_lock lock{mutex};
        return authorization[path];
    }

    std::string Param(const std::string& path, const std::string& key) {
        std::scoped_lock lock{mutex};
        const auto& form = forms[path];
        const auto it = form.find(key);
        return it == form.end() ? std::string{} : it->second;
    }

private:
    struct Canned {
        int status;
        std::string body;
    };

    std::filesystem::path directory;
    std::filesystem::path previous_config;
    std::filesystem::path previous_cache;
    httplib::Server server;
    std::thread thread;
    int port{};
    std::mutex mutex;
    std::map<std::string, std::deque<Canned>> replies;
    std::map<std::string, int> hits;
    std::map<std::string, std::string> authorization;
    std::map<std::string, httplib::Params> forms;
};

constexpr const char* ClientId = "nxc_test";
constexpr const char* TokenPath = "/api/oauth/token";
constexpr const char* NexTokenPath = "/api/nex-token";
constexpr const char* NewGameToken = R"({"nex_token":"nx2.new","pid":1,"username":"tester",)"
                                     R"("friend_code":"SW-0000-0000-0000","expires_in":86400})";
constexpr const char* NewAccessToken =
    R"({"access_token":"access.new","refresh_token":"refresh.new","expires_in":3600})";

} // namespace

TEST_CASE("Nextendo: the game token is renewed ahead of its expiry", "[common]") {
    FakeNextendo fake;
    fake.Answer(NexTokenPath, 200, NewGameToken);

    Nextendo::SaveAccount(SignedIn(2 * 60 * 60, 30 * 60));
    REQUIRE(Nextendo::EnsureGameToken(ClientId)->nex_token == "nx2.old");
    REQUIRE(fake.TotalHits() == 0);

    Nextendo::SaveAccount(SignedIn(Nextendo::RenewAheadSeconds - 60, 30 * 60));
    const auto account = Nextendo::EnsureGameToken(ClientId);
    REQUIRE(account->nex_token == "nx2.new");
    REQUIRE(account->nex_expires >= Now() + 86400 - 5);
    REQUIRE(fake.Hits(NexTokenPath) == 1);
    REQUIRE(fake.Hits(TokenPath) == 0);
    REQUIRE(fake.Authorization(NexTokenPath) == "Bearer access.old");
}

TEST_CASE("Nextendo: an expired access token is refreshed before the game token", "[common]") {
    FakeNextendo fake;
    fake.Answer(TokenPath, 200, NewAccessToken);
    fake.Answer(NexTokenPath, 200, NewGameToken);

    Nextendo::SaveAccount(SignedIn(60, -10));
    const auto account = Nextendo::EnsureGameToken(ClientId);
    REQUIRE(account->access_token == "access.new");
    REQUIRE(account->refresh_token == "refresh.new");
    REQUIRE(account->nex_token == "nx2.new");
    REQUIRE(fake.Param(TokenPath, "grant_type") == "refresh_token");
    REQUIRE(fake.Authorization(NexTokenPath) == "Bearer access.new");
}

TEST_CASE("Nextendo: invalid_grant signs out instead of retrying", "[common]") {
    FakeNextendo fake;
    fake.Answer(TokenPath, 400, R"({"error":"invalid_grant"})");

    Nextendo::SaveAccount(SignedIn(60, -10));
    REQUIRE_FALSE(Nextendo::EnsureGameToken(ClientId).has_value());
    REQUIRE(Nextendo::LastSignOutReason() == Nextendo::SignOutReason::SessionExpired);
    REQUIRE_FALSE(Nextendo::EnsureGameToken(ClientId).has_value());
    REQUIRE(fake.TotalHits() == 1);
}

TEST_CASE("Nextendo: 403 asks for a sign-in with game.matchmaking", "[common]") {
    FakeNextendo fake;
    fake.Answer(NexTokenPath, 403, R"({"error":"insufficient_scope"})");

    Nextendo::SaveAccount(SignedIn(60, 30 * 60));
    REQUIRE_FALSE(Nextendo::EnsureGameToken(ClientId).has_value());
    REQUIRE(Nextendo::LastSignOutReason() == Nextendo::SignOutReason::MissingScope);
    REQUIRE(fake.Hits(TokenPath) == 0);
    REQUIRE(fake.Hits(NexTokenPath) == 1);
}

TEST_CASE("Nextendo: 401 refreshes the access token once", "[common]") {
    SECTION("and the new one works") {
        FakeNextendo fake;
        fake.Answer(NexTokenPath, 401, "{}");
        fake.Answer(NexTokenPath, 200, NewGameToken);
        fake.Answer(TokenPath, 200, NewAccessToken);

        Nextendo::SaveAccount(SignedIn(60, 30 * 60));
        REQUIRE(Nextendo::EnsureGameToken(ClientId)->nex_token == "nx2.new");
        REQUIRE(fake.Hits(TokenPath) == 1);
        REQUIRE(fake.Hits(NexTokenPath) == 2);
    }
    SECTION("and signs out when the new one is refused too") {
        FakeNextendo fake;
        fake.Answer(NexTokenPath, 401, "{}");
        fake.Answer(TokenPath, 200, NewAccessToken);

        Nextendo::SaveAccount(SignedIn(60, 30 * 60));
        REQUIRE_FALSE(Nextendo::EnsureGameToken(ClientId).has_value());
        REQUIRE(Nextendo::LastSignOutReason() == Nextendo::SignOutReason::SessionRevoked);
        REQUIRE(fake.Hits(TokenPath) == 1);
        REQUIRE(fake.Hits(NexTokenPath) == 2);
    }
}

TEST_CASE("Nextendo: a server error keeps the account and backs off", "[common]") {
    FakeNextendo fake;
    fake.Answer(NexTokenPath, 500, "{}");

    Nextendo::SaveAccount(SignedIn(60, 30 * 60));
    REQUIRE(Nextendo::EnsureGameToken(ClientId)->nex_token == "nx2.old");
    REQUIRE(Nextendo::EnsureGameToken(ClientId)->nex_token == "nx2.old");
    REQUIRE(fake.Hits(NexTokenPath) == 1);
}

TEST_CASE("Nextendo: session heartbeat", "[common]") {
    SECTION("a refused game token is renewed at the next check") {
        FakeNextendo fake;
        fake.Answer("/api/nex-session", 401, "{}");
        fake.Answer(NexTokenPath, 200, NewGameToken);

        Nextendo::SaveAccount(SignedIn(2 * 60 * 60, 30 * 60));
        REQUIRE(Nextendo::KeepSessionAlive(ClientId) == Nextendo::SignOutReason::None);
        REQUIRE(fake.Hits(NexTokenPath) == 0);
        REQUIRE(Nextendo::EnsureGameToken(ClientId)->nex_token == "nx2.new");
    }
    SECTION("a session ended on the site signs out") {
        FakeNextendo fake;
        fake.Answer("/api/nex-session", 200, "{}");
        fake.Answer("/api/session-status", 200, R"({"revoked":true})");

        Nextendo::SaveAccount(SignedIn(2 * 60 * 60, 30 * 60));
        REQUIRE(Nextendo::KeepSessionAlive(ClientId) == Nextendo::SignOutReason::SessionRevoked);
        REQUIRE(fake.Authorization("/api/nex-session") == "Bearer nx2.old");
        REQUIRE_FALSE(Nextendo::LoadAccount().has_value());
    }
    SECTION("a refresh refused during the heartbeat is reported") {
        FakeNextendo fake;
        fake.Answer(TokenPath, 400, R"({"error":"invalid_grant"})");

        Nextendo::SaveAccount(SignedIn(60, -10));
        REQUIRE(Nextendo::KeepSessionAlive(ClientId) == Nextendo::SignOutReason::SessionExpired);
    }
}

TEST_CASE("Nextendo: sign-out revokes the refresh token", "[common]") {
    FakeNextendo fake;
    fake.Answer("/api/oauth/revoke", 200, R"({"ok":true})");

    Nextendo::RevokeToken(ClientId, "refresh.old");
    REQUIRE(fake.Param("/api/oauth/revoke", "token") == "refresh.old");
}

TEST_CASE("Nextendo: requests stay within the per-minute budget", "[common]") {
    FakeNextendo fake;
    fake.Answer("/api/nex-session", 200, "{}");
    fake.Answer("/api/session-status", 200, R"({"revoked":false})");

    Nextendo::SaveAccount(SignedIn(2 * 60 * 60, 30 * 60));
    for (int i = 0; i < 12; ++i) {
        void(Nextendo::KeepSessionAlive(ClientId));
    }
    REQUIRE(fake.TotalHits() == static_cast<int>(Nextendo::RequestsPerMinute));
}

TEST_CASE("Nextendo: sign-in takes a free port when the callback port is busy", "[common]") {
    FakeNextendo fake;
    fake.Answer(TokenPath, 200, NewAccessToken);
    fake.Answer(NexTokenPath, 200, NewGameToken);
    fake.Answer("/api/online-status", 200, R"({"allow":false,"reason":"discord_unlinked"})");

    httplib::Server blocker;
    blocker.set_socket_options([](socket_t) {});
    const bool blocked = blocker.bind_to_port("127.0.0.1", Nextendo::CallbackPort);

    std::thread browser;
    std::string redirect_port;
    const auto open_url = [&](const std::string& url) {
        const auto field = [&url](std::string_view key) {
            const auto start = url.find(key) + key.size();
            return url.substr(start, url.find('&', start) - start);
        };
        const std::string redirect = field("&redirect_uri=http%3A%2F%2F127.0.0.1%3A");
        redirect_port = redirect.substr(0, redirect.find('%'));
        browser = std::thread{[port = std::stoi(redirect_port), state = field("&state=")] {
            httplib::Client client{"127.0.0.1", port};
            void(client.Get("/callback?code=abc&state=" + state));
        }};
    };
    const auto result = Nextendo::SignIn(ClientId, open_url, std::stop_token{});
    if (browser.joinable()) {
        browser.join();
    }

    INFO(result.error);
    REQUIRE(result.ok);
    if (blocked) {
        REQUIRE(redirect_port != std::to_string(Nextendo::CallbackPort));
    }
    REQUIRE(fake.Param(TokenPath, "redirect_uri") ==
            "http://127.0.0.1:" + redirect_port + "/callback");
    REQUIRE(result.account.nex_token == "nx2.new");
    REQUIRE(fake.Authorization("/api/online-status") == "Bearer nx2.new");
    REQUIRE(result.online_status.has_value());
    REQUIRE(result.online_status->reason == "discord_unlinked");
}

TEST_CASE("Nextendo: the BAAS user id is the account id only in Animal Crossing", "[common]") {
    REQUIRE(Nextendo::ParseBaasNsa("0123456789abcDEF") == 0x0123456789ABCDEF);
    REQUIRE(Nextendo::ParseBaasNsa("123456789abcdef") == 0);
    REQUIRE(Nextendo::ParseBaasNsa("0123456789abcdeg") == 0);
    REQUIRE(Nextendo::ParseBaasNsa("") == 0);

    Nextendo::Account account{.pid = 1800000001, .baas_nsa = 0x0123456789ABCDEF};
    REQUIRE(Nextendo::NetworkServiceAccountId(Nextendo::AnimalCrossingNewHorizons, account) ==
            0x0123456789ABCDEF);
    REQUIRE(Nextendo::NetworkServiceAccountId(0x0100152000022000, account) == 1800000001);
    account.baas_nsa = 0;
    REQUIRE(Nextendo::NetworkServiceAccountId(Nextendo::AnimalCrossingNewHorizons, account) ==
            1800000001);
}

TEST_CASE("Nextendo: a game token renewal fetches the BAAS user id once", "[common]") {
    FakeNextendo fake;
    fake.Answer(NexTokenPath, 200, NewGameToken);
    fake.Answer("/api/me", 200, R"({"pid":1,"username":"tester","nsa":"0123456789abcdef"})");

    Nextendo::SaveAccount(SignedIn(Nextendo::RenewAheadSeconds - 60, 30 * 60));
    REQUIRE(Nextendo::EnsureGameToken(ClientId)->baas_nsa == 0x0123456789ABCDEF);
    REQUIRE(fake.Authorization("/api/me") == "Bearer access.old");

    auto renewed = *Nextendo::LoadAccount();
    renewed.nex_expires = Now() + Nextendo::RenewAheadSeconds - 60;
    Nextendo::SaveAccount(renewed);
    REQUIRE(Nextendo::EnsureGameToken(ClientId)->baas_nsa == 0x0123456789ABCDEF);
    REQUIRE(fake.Hits(NexTokenPath) == 2);
    REQUIRE(fake.Hits("/api/me") == 1);
}

namespace {

std::string Base64(std::span<const u8> data) {
    std::string out(4 * ((data.size() + 2) / 3) + 1, '\0');
    const int size = EVP_EncodeBlock(reinterpret_cast<unsigned char*>(out.data()), data.data(),
                                     static_cast<int>(data.size()));
    out.resize(static_cast<std::size_t>(size));
    return out;
}

std::string Timestamp(s64 time) {
    const std::time_t t = static_cast<std::time_t>(time);
    std::array<char, 32> text{};
    std::strftime(text.data(), text.size(), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
    return text.data();
}

// Signs catalogues with a key made for the test, the way nextendo.network signs its own.
class CatalogSigner {
public:
    CatalogSigner() {
        EVP_PKEY_CTX* const context = EVP_PKEY_CTX_new_id(EVP_PKEY_ED25519, nullptr);
        REQUIRE(context != nullptr);
        REQUIRE(EVP_PKEY_keygen_init(context) == 1);
        REQUIRE(EVP_PKEY_keygen(context, &key) == 1);
        EVP_PKEY_CTX_free(context);
    }

    ~CatalogSigner() {
        EVP_PKEY_free(key);
    }

    std::string PublicKey() const {
        std::array<u8, 32> raw{};
        std::size_t size = raw.size();
        REQUIRE(EVP_PKEY_get_raw_public_key(key, raw.data(), &size) == 1);
        return Base64(raw);
    }

    std::string Sign(std::string_view payload, std::string_view key_id = "catalog-key-1") const {
        std::array<u8, 64> signature{};
        std::size_t size = signature.size();
        EVP_MD_CTX* const context = EVP_MD_CTX_new();
        REQUIRE(EVP_DigestSignInit(context, nullptr, nullptr, nullptr, key) == 1);
        REQUIRE(EVP_DigestSign(context, signature.data(), &size,
                               reinterpret_cast<const u8*>(payload.data()), payload.size()) == 1);
        EVP_MD_CTX_free(context);
        return fmt::format(R"({{"key_id":"{}","payload":"{}","signature":"{}"}})", key_id,
                           Base64({reinterpret_cast<const u8*>(payload.data()), payload.size()}),
                           Base64(signature));
    }

private:
    EVP_PKEY* key{};
};

std::string CatalogPayload(s64 expires_at, std::string_view type = "game_catalog",
                           int revision = 3) {
    return fmt::format(
        R"({{"document_type":"{}","schema_version":1,"revision":{},"issued_at":"{}",)"
        R"("expires_at":"{}","games":[)"
        R"({{"title_ids":["0100152000022000"],"name":"Mario Kart 8 Deluxe","enabled":true,)"
        R"("protocol":"nex","required_capabilities":["versions.exact.v1"],)"
        R"("versions":{{"policy":"exact","accepted":["4.0.1"],"recommended":"4.0.1"}}}},)"
        R"({{"title_ids":["0100f8f0000a2000","01003BC0000A0000"],"name":"Splatoon 2",)"
        R"("enabled":false,"protocol":"nex","required_capabilities":[]}},)"
        R"({{"title_ids":["0100b0c013912000"],"name":"Among Us","enabled":true,)"
        R"("protocol":"other","required_capabilities":["versions.any.v1"],)"
        R"("versions":{{"policy":"any"}}}}]}})",
        type, revision, Timestamp(expires_at - 86400), Timestamp(expires_at));
}

} // namespace

TEST_CASE("Nextendo: the game catalogue is checked before it is used", "[common]") {
    const CatalogSigner signer;
    const s64 now = Now();
    const auto valid = signer.Sign(CatalogPayload(now + 3600));

    const auto catalog = Nextendo::ParseGameCatalog(valid, signer.PublicKey(), now);
    REQUIRE(catalog.has_value());
    REQUIRE(catalog->revision == 3);
    REQUIRE(catalog->titles.size() == 4);
    REQUIRE(catalog->titles.at(0x01003BC0000A0000).enabled == false);

    REQUIRE_FALSE(Nextendo::ParseGameCatalog(valid, Nextendo::GameCatalogKey, now).has_value());
    REQUIRE_FALSE(Nextendo::ParseGameCatalog(signer.Sign(CatalogPayload(now + 3600), "other-key"),
                                             signer.PublicKey(), now)
                      .has_value());
    REQUIRE_FALSE(Nextendo::ParseGameCatalog(valid, signer.PublicKey(), now + 3600).has_value());
    REQUIRE_FALSE(Nextendo::ParseGameCatalog(signer.Sign(CatalogPayload(now + 3600, "npln")),
                                             signer.PublicKey(), now)
                      .has_value());
    REQUIRE_FALSE(
        Nextendo::ParseGameCatalog(signer.Sign(CatalogPayload(now + 3600, "game_catalog", 0)),
                                   signer.PublicKey(), now)
            .has_value());

    std::string tampered = valid;
    tampered[tampered.find("payload") + 12] ^= 1;
    REQUIRE_FALSE(Nextendo::ParseGameCatalog(tampered, signer.PublicKey(), now).has_value());
}

TEST_CASE("Nextendo: game versions follow the catalogue's exact policy", "[common]") {
    const CatalogSigner signer;
    const auto catalog = *Nextendo::ParseGameCatalog(signer.Sign(CatalogPayload(Now() + 3600)),
                                                     signer.PublicKey(), Now());
    using Kind = Nextendo::VersionVerdict::Kind;
    constexpr u64 Mk8dx = 0x0100152000022000;

    REQUIRE(Nextendo::CheckGameVersion(catalog, Mk8dx, "4.0.1")->kind == Kind::Accepted);
    REQUIRE(Nextendo::CheckGameVersion(catalog, Mk8dx, "4.0.1.0")->kind == Kind::Accepted);
    const auto old = Nextendo::CheckGameVersion(catalog, Mk8dx, "4.0.0");
    REQUIRE(old->kind == Kind::UpdateRequired);
    REQUIRE(old->recommended == "4.0.1");
    REQUIRE(Nextendo::CheckGameVersion(catalog, Mk8dx, "4.0.1.1")->kind == Kind::UpdateRequired);
    REQUIRE(Nextendo::CheckGameVersion(catalog, 0x0100F8F0000A2000, "5.5.2")->kind ==
            Kind::Disabled);
    REQUIRE(Nextendo::CheckGameVersion(catalog, 0x0100B0C013912000, "2026.19.0")->kind ==
            Kind::Accepted);
    REQUIRE_FALSE(Nextendo::CheckGameVersion(catalog, 0x01006F8002326000, "3.0.3").has_value());
    REQUIRE_FALSE(Nextendo::CheckGameVersion(catalog, Mk8dx, "").has_value());
}

TEST_CASE("Nextendo: the game catalogue is cached until it expires", "[common]") {
    FakeNextendo fake;
    const CatalogSigner signer;
    Settings::values.nextendo_enabled = true;

    fake.Answer("/service-catalog/v1/game-catalog", 200, "not signed");
    fake.Answer("/service-catalog/v1/game-catalog", 200, signer.Sign(CatalogPayload(Now() + 3600)));
    REQUIRE_FALSE(Nextendo::LoadGameCatalog(ClientId, signer.PublicKey()).has_value());
    REQUIRE(Nextendo::LoadGameCatalog(ClientId, signer.PublicKey())->revision == 3);
    REQUIRE(Nextendo::LoadGameCatalog(ClientId, signer.PublicKey())->revision == 3);
    REQUIRE(fake.Hits("/service-catalog/v1/game-catalog") == 2);

    Settings::values.nextendo_enabled = false;
    REQUIRE_FALSE(Nextendo::LoadGameCatalog(ClientId, signer.PublicKey()).has_value());
}
