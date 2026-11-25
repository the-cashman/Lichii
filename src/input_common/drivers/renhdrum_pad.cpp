// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "input_common/drivers/renhdrum_pad.h"

#ifdef ENABLE_RENHDRUM

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <renhdrum.h>

#ifdef RENHDRUM_HAVE_PACKAGED_PATH
#include "renhdrum_library_path.h"
#endif

#include "common/dynamic_library.h"
#include "common/logging.h"
#include "common/settings.h"
#include "common/thread.h"

#ifdef RENHDRUM_SDL_HDRUMBLE
// Added by .patch/sdl3-hdrumble/0002-hdrumble-idle-write.patch (SDL3 is linked statically). Each
// returns 1 when the mode is active, 0 when SDL refused it, -1 for a pad that is not a HIDAPI
// Switch pad.
extern "C" {
int SDLCALL SDL_HDRumbleSetNoQueue(SDL_Joystick* joystick, int enable, int gap_ms, int pause_ms);
int SDLCALL SDL_HDRumbleSetNoQueueBluetooth(SDL_Joystick* joystick, int enable, int gap_ms);
int SDLCALL SDL_HDRumbleSetBluetoothAfterInput(SDL_Joystick* joystick, int enable);
}
#endif


namespace {

/// RenHDrum 0.5.0's settings table, for libraries that predate `renhdrum_options` (and for
/// builds without the library). The dialog binds every row to this build's settings either way,
/// so an old library only loses the live wording, never an option.
std::vector<InputCommon::RenHDrumOption> FallbackRenHDrumOptions() {
    // k: key, l: label, h: help, c: choices ("a|b"), kind, min, max, step, def,
    // dev: bits (1 Switch, 2 DualSense, 4 Xbox), tier, link, req: requires row index
    using InputCommon::RenHDrumOption;
    auto row = [](const char* k, const char* l, const char* h, const char* c, int kind, int mn,
                  int mx, int st, int df, unsigned dev, int tier, int link, int req) {
        RenHDrumOption o;
        o.key = k;
        o.label = l;
        o.help = h;
        if (c && *c) {
            std::string choices{c};
            std::size_t start = 0;
            while (true) {
                const std::size_t bar = choices.find('|', start);
                o.choices.push_back(choices.substr(start, bar - start));
                if (bar == std::string::npos) {
                    break;
                }
                start = bar + 1;
            }
        }
        o.kind = kind;
        o.min = mn;
        o.max = mx;
        o.step = st;
        o.def = df;
        o.devices = dev;
        o.tier = tier;
        o.link = link;
        o.requires_row = req;
        return o;
    };
    constexpr unsigned sw = 1, ds = 2, xb = 4, all = sw | ds | xb | 8;
    return {
        row("enable_hd_rumble", "HD Rumble",
            "Plays the game's HD Rumble in full detail. Off: the standard rumble.", "", 0, 0, 1,
            1, 1, all, 0, 0, -1),
        row("switch_bt_mode", "Bluetooth mode",
            "Detailed (5 ms): every detail, like wired. Compatible: works on every Bluetooth "
            "adapter; used automatically when Detailed cannot run.",
            "Detailed (5 ms)|Compatible", 2, 0, 1, 1, 0, sw, 0, 2, -1),
        row("dualsense_haptics", "Haptics",
            "Rumble through the two haptic voice coils. Off: the standard rumble.", "", 0, 0, 1, 1,
            1, ds, 0, 0, -1),
        row("dualsense_hd_haptics", "HD Haptics",
            "HD Rumble upgraded for DualSense. The controller speaker and the L2/R2 triggers "
            "react to the rumble. Off = standard HD Rumble.",
            "", 0, 0, 1, 1, 1, ds, 0, 0, 2),
        row("dualsense_volume", "Strength", "Lower it if the rumble feels too strong.", "", 1, 10,
            100, 5, 100, ds, 0, 0, 2),
        row("xbox_enabled", "HD Rumble on Xbox",
            "The game's HD Rumble translated for the Xbox motors. Off: the standard rumble.",
            "", 0, 0, 1, 1, 1, xb, 0, 0, -1),
        row("gain", "Intensity", "Overall rumble strength.", "", 1, 25, 1000, 5, 100, all, 1, 0,
            -1),
        row("low_boost", "Lift weak rumble", "Makes faint rumble easier to feel.", "", 1, 0, 100,
            5, 0, all, 1, 0, -1),
        row("usb_no_queue", "Send rumble without queuing", "", "", 0, 0, 1, 1, 1, sw, 1, 1, -1),
        row("envelope_5ms", "Detailed 5 ms rumble", "", "", 0, 0, 1, 1, 1, sw, 1, 1, -1),
        row("frequency_in_packet", "Accurate pitch, low band", "", "", 0, 0, 1, 1, 1, sw, 1, 1,
            -1),
        row("frequency_both_bands", "Accurate pitch, both bands", "", "", 0, 0, 1, 1, 1, sw, 1, 1,
            -1),
        row("crisp_notes", "Accent each note", "", "", 0, 0, 1, 1, 1, sw, 1, 1, -1),
        row("usb_interval_ms", "Send interval (when detailed rumble is off)",
            "Used only while Detailed 5 ms rumble is off.", "10 ms|15 ms|20 ms|30 ms", 2, 0, 3, 1,
            1, sw, 1, 1, -1),
        row("usb_auto_slow", "Slow down during heavy rumble", "", "", 0, 0, 1, 1, 1, sw, 1, 1,
            -1),
        row("bt_after_input", "Send rumble between button reads", "", "", 0, 0, 1, 1, 1, sw, 1, 2,
            -1),
        row("bt_crisp_notes", "Accent each note", "", "", 0, 0, 1, 1, 1, sw, 1, 2, -1),
        row("dualsense_crisp_notes", "Accent each note", "", "", 0, 0, 1, 1, 1, ds, 1, 0, -1),
        row("xbox_mode", "Mode", "", "RenHDrum|Direct", 2, 0, 1, 1, 0, xb, 1, 0, -1),
        row("xbox_strength", "Motor strength", "100% = as the game sends it. Raise it if the "
                                              "rumble feels weak.",
            "", 1, 0, 1000, 10, 100, xb, 1, 0, 5),
        row("xbox_detail", "Detail on main motor", "", "", 1, 0, 100, 5, 50, xb, 1, 0, 18),
        row("xbox_accents", "Hit accents", "", "", 1, 0, 100, 5, 50, xb, 1, 0, 18),
        row("xbox_trigger_mix", "Trigger rumble", "", "", 1, 0, 100, 5, 20, xb, 1, 0, 18),
    };
}

} // namespace

namespace InputCommon {

namespace {

// The RenHDrum C ABI, resolved from the shared library at runtime: nothing to link, and a build
// without the library just keeps the standard rumble.
struct RenHDrumApi {
    decltype(&renhdrum_engine_create) engine_create{};
    decltype(&renhdrum_engine_destroy) engine_destroy{};
    decltype(&renhdrum_engine_apply_settings) engine_apply_settings{};
    decltype(&renhdrum_engine_push_half_sample) engine_push_half_sample{};
    decltype(&renhdrum_engine_tick) engine_tick{};
    decltype(&renhdrum_settings_resolve) settings_resolve{};
    // Optional (RenHDrum 0.4.0+): the resolver that reads no IPC and no flag files. Preferred,
    // because this host has no Brontide-App config and the plain resolver would look for one.
    decltype(&renhdrum_settings_resolve_pure) settings_resolve_pure{};
    // Optional (RenHDrum 0.4.0+): the Switch 2 Mode 2 translation, chosen by the guest's Mode.
    decltype(&renhdrum_set_nx2_mode) set_nx2_mode{};
    decltype(&renhdrum_dualsense_dsp_create) dsp_create{};
    decltype(&renhdrum_dualsense_dsp_destroy) dsp_destroy{};
    decltype(&renhdrum_dualsense_dsp_set_half_target) dsp_set_half_target{};
    decltype(&renhdrum_dualsense_dsp_fill_f32) dsp_fill_f32{};
    // Optional (RenHDrum 0.2.1+): DualSense haptics over Bluetooth
    decltype(&renhdrum_dualsense_dsp_build_bt_report) dsp_build_bt_report{};
    // Optional (RenHDrum 0.5.0+): DualSense HD Haptics -- the speaker and the trigger kick
    decltype(&renhdrum_dualsense_dsp_set_speaker) dsp_set_speaker{};
    decltype(&renhdrum_hd_haptics_speaker_block) hd_speaker_block{};
    decltype(&renhdrum_hd_haptics_trigger_block) hd_trigger_block{};
    decltype(&renhdrum_tri_trigger_create) tri_create{};
    decltype(&renhdrum_tri_trigger_destroy) tri_destroy{};
    decltype(&renhdrum_tri_trigger_update) tri_update{};
    // Strictly optional even when the rest is there: a hint the core works without
    decltype(&renhdrum_tri_trigger_note_action) tri_note_action{};
    // Optional (RenHDrum 0.5.0+): the combined rumble + trigger + speaker report 0x36
    decltype(&renhdrum_dualsense_dsp_build_bt36_report) dsp_build_bt36_report{};
    // Optional (RenHDrum 0.5.0+): the settings table the dialog builds its rows from
    decltype(&renhdrum_options) options{};
    bool HasTriTrigger() const {
        return tri_create && tri_destroy && tri_update;
    }
    bool HasHdHaptics() const {
        return HasTriTrigger() && dsp_set_speaker && hd_speaker_block && hd_trigger_block;
    }
    // Optional (RenHDrum 0.2.0+): Xbox-style controllers
    decltype(&renhdrum_xbox_rumble_create) xbox_create{};
    decltype(&renhdrum_xbox_rumble_destroy) xbox_destroy{};
    decltype(&renhdrum_xbox_rumble_push_half_sample) xbox_push_half_sample{};
    decltype(&renhdrum_xbox_rumble_tick) xbox_tick{};
    decltype(&renhdrum_xbox_rumble_resend) xbox_resend{};
    bool HasXbox() const {
        return xbox_create && xbox_destroy && xbox_push_half_sample && xbox_tick && xbox_resend;
    }

    Common::DynamicLibrary library;
    bool loaded{};
};

const RenHDrumApi& Api() {
    static const RenHDrumApi api = [] {
        RenHDrumApi a;
#if defined(_WIN32)
        constexpr const char* name = "renhdrum.dll";
#elif defined(__APPLE__)
        constexpr const char* name = "librenhdrum.dylib";
#else
        constexpr const char* name = "librenhdrum.so";
#endif
        // dlopen does not look in the executable's directory the way LoadLibrary does on Windows,
        // so "next to the emulator" only works if we name it ourselves. Try, in order: the
        // RENHDRUM_LIBRARY override, the executable's directory, then the loader's search path.
        std::vector<std::string> candidates;
        if (const char* override_path = std::getenv("RENHDRUM_LIBRARY");
            override_path != nullptr && *override_path != '\0') {
            candidates.emplace_back(override_path);
        }
#ifdef RENHDRUM_HAVE_PACKAGED_PATH
        // Where the packager installed it (nix/package.nix): a store path, so it is referenced and
        // GC-safe, and nobody has to set an environment variable. An explicit override still wins.
        candidates.emplace_back(RENHDRUM_LIBRARY_PATH);
#endif
        // SDL3 owns the string; it must not be freed.
        if (const char* base_path = SDL_GetBasePath()) {
            candidates.emplace_back(std::string{base_path} + name);
        }
        candidates.emplace_back(name);

        std::string opened;
        for (const auto& candidate : candidates) {
            a.library = Common::DynamicLibrary(candidate.c_str());
            if (!a.library.IsOpen()) {
                continue;
            }
            // An opened candidate is only accepted when every required symbol resolves: a
            // custom build that predates the ABI this emulator compiled against must not
            // stop the search for the packaged library later in the list.
            a.loaded =
                a.library.GetSymbol("renhdrum_engine_create", &a.engine_create) &&
                a.library.GetSymbol("renhdrum_engine_destroy", &a.engine_destroy) &&
                a.library.GetSymbol("renhdrum_engine_apply_settings", &a.engine_apply_settings) &&
                a.library.GetSymbol("renhdrum_engine_push_half_sample",
                                     &a.engine_push_half_sample) &&
                a.library.GetSymbol("renhdrum_engine_tick", &a.engine_tick) &&
                a.library.GetSymbol("renhdrum_settings_resolve", &a.settings_resolve) &&
                a.library.GetSymbol("renhdrum_dualsense_dsp_create", &a.dsp_create) &&
                a.library.GetSymbol("renhdrum_dualsense_dsp_destroy", &a.dsp_destroy) &&
                a.library.GetSymbol("renhdrum_dualsense_dsp_set_half_target",
                                     &a.dsp_set_half_target) &&
                a.library.GetSymbol("renhdrum_dualsense_dsp_fill_f32", &a.dsp_fill_f32);
            if (!a.loaded) {
                LOG_WARNING(Input,
                            "RenHDrum: {} is missing functions, trying the next candidate",
                            candidate);
                a.library = Common::DynamicLibrary{};
                continue;
            }
            opened = candidate;
            break;
        }
        if (opened.empty()) {
            LOG_INFO(Input, "RenHDrum: no usable {} found, using the standard rumble", name);
            return a;
        }
        Settings::values.renhdrum_available.SetValue(a.loaded);
        if (a.loaded) {
            // Optional: a missing symbol leaves it null
            (void)a.library.GetSymbol("renhdrum_settings_resolve_pure", &a.settings_resolve_pure);
            (void)a.library.GetSymbol("renhdrum_set_nx2_mode", &a.set_nx2_mode);
            (void)a.library.GetSymbol("renhdrum_dualsense_dsp_build_bt_report",
                                      &a.dsp_build_bt_report);
            (void)a.library.GetSymbol("renhdrum_dualsense_dsp_set_speaker", &a.dsp_set_speaker);
            (void)a.library.GetSymbol("renhdrum_hd_haptics_speaker_block", &a.hd_speaker_block);
            (void)a.library.GetSymbol("renhdrum_hd_haptics_trigger_block", &a.hd_trigger_block);
            (void)a.library.GetSymbol("renhdrum_tri_trigger_create", &a.tri_create);
            (void)a.library.GetSymbol("renhdrum_tri_trigger_destroy", &a.tri_destroy);
            (void)a.library.GetSymbol("renhdrum_tri_trigger_update", &a.tri_update);
            (void)a.library.GetSymbol("renhdrum_tri_trigger_note_action", &a.tri_note_action);
            (void)a.library.GetSymbol("renhdrum_dualsense_dsp_build_bt36_report",
                                      &a.dsp_build_bt36_report);
            (void)a.library.GetSymbol("renhdrum_options", &a.options);
            (void)a.library.GetSymbol("renhdrum_xbox_rumble_create", &a.xbox_create);
            (void)a.library.GetSymbol("renhdrum_xbox_rumble_destroy", &a.xbox_destroy);
            (void)a.library.GetSymbol("renhdrum_xbox_rumble_push_half_sample",
                                      &a.xbox_push_half_sample);
            (void)a.library.GetSymbol("renhdrum_xbox_rumble_tick", &a.xbox_tick);
            (void)a.library.GetSymbol("renhdrum_xbox_rumble_resend", &a.xbox_resend);
            LOG_INFO(Input, "RenHDrum: loaded {}", opened);
        } else {
            LOG_WARNING(Input, "RenHDrum: {} is missing functions, using the standard rumble",
                        opened);
        }
        return a;
    }();
    return api;
}

constexpr Uint16 NintendoVendorId = 0x057E;
constexpr Uint16 SonyVendorId = 0x054C;
constexpr Uint16 MicrosoftVendorId = 0x045E;
constexpr std::array<Uint8, 10> NeutralPacket{0x10, 0, 0x00, 0x01, 0x40, 0x40, 0x00, 0x01, 0x40, 0x40};
constexpr int DualSenseAudioMaxAttempts = 10;
// SDL no-queue gap while the 5 ms envelope runs (Ryuhdr AmFmNoQueueGapMs)
[[maybe_unused]] constexpr int NoQueueEnvelopeGapMs = 10;

// HD Haptics (RenHDrum 0.5.0): the 47-byte DualSense effects blocks come from the core, ready to
// hand to SDL_SendJoystickEffect. Only used once Api().HasHdHaptics() is true.
using Ds5Block = std::array<Uint8, RENHDRUM_DS5_EFFECTS_SIZE>;

Ds5Block HdSpeakerBlock(bool on) {
    Ds5Block block{};
    Api().hd_speaker_block(on ? 1 : 0, block.data());
    return block;
}

/// `effect` null = trigger off.
Ds5Block HdTriggerBlock(bool is_right, const std::array<Uint8, 11>* effect) {
    Ds5Block block{};
    Api().hd_trigger_block(is_right ? 1 : 0, effect ? effect->data() : nullptr, block.data());
    return block;
}

bool IsSwitchHdController(Uint16 vid, Uint16 pid) {
    return vid == NintendoVendorId && (pid == 0x2009 || pid == 0x2006 || pid == 0x2007);
}

/// Xbox One / Series and Xbox 360 style pads, Microsoft's or not (GameSir, 8BitDo, PowerA...).
/// RenHDrum's converter was tuned on an Xbox One S; others get the same four-motor mix, and
/// a pad whose path has no trigger motors (XInput) keeps only the grips.
bool IsXboxStyle(SDL_Joystick* joystick, SDL_Gamepad* gamepad) {
    if (gamepad) {
        const SDL_GamepadType type = SDL_GetGamepadType(gamepad);
        if (type == SDL_GAMEPAD_TYPE_XBOXONE || type == SDL_GAMEPAD_TYPE_XBOX360) {
            return true;
        }
    }
#ifdef _WIN32
    // With raw input on, SDL types a third-party XInput pad by its vendor ID and may not call it
    // an Xbox controller. Windows marks every XInput-capable HID device with "IG_" in its path.
    const char* path = SDL_GetJoystickPath(joystick);
    if (path && (std::strstr(path, "IG_") != nullptr || std::strstr(path, "ig_") != nullptr)) {
        return true;
    }
#else
    (void)joystick;
#endif
    return false;
}

bool IsDualSense(Uint16 vid, Uint16 pid) {
    return vid == SonyVendorId && (pid == 0x0CE6 || pid == 0x0DF2);
}

struct UserSettings {
    bool hd_enabled{true};
    int usb_interval_ms{15};
    bool envelope_5ms{true};
    bool frequency_in_packet{true};
    bool crisp_notes{true};
    int bluetooth_level{3};
    bool bt_after_input{true};
    bool bt_fast{true};
    bool bt_crisp_notes{true};
    // The 0.5.0 options-table rows that had no dedicated field before
    int bt_mode{0};               // 0 Detailed (5 ms), 1 Compatible
    bool usb_no_queue{true};
    bool frequency_both_bands{true};
    bool usb_auto_slow{true};
    bool dualsense_enabled{true};
    int dualsense_volume{100};
    bool dualsense_crisp_notes{true};
    bool dualsense_hd_haptics{true};
    int dualsense_speaker_volume{60};
    bool xbox_enabled{true};
    int xbox_strength{100};
    bool xbox_direct{false};
    int xbox_detail_mix{50};
    int xbox_hit_accents{50};
    int xbox_trigger_mix{20};
    int gain{100};
    int low_boost{0};
    bool operator==(const UserSettings&) const = default;
};

UserSettings ReadUserSettings() {
    const auto& s = Settings::values;
    return {
        .hd_enabled = s.hd_rumble_enabled.GetValue(),
        .usb_interval_ms = std::clamp(s.hd_rumble_usb_interval_ms.GetValue(), 5, 100),
        .envelope_5ms = s.hd_rumble_envelope_5ms.GetValue(),
        .frequency_in_packet = s.hd_rumble_frequency_in_packet.GetValue(),
        .crisp_notes = s.hd_rumble_crisp_notes.GetValue(),
        .bluetooth_level = std::clamp(s.hd_rumble_bluetooth_level.GetValue(), 1, 3),
        .bt_after_input = s.hd_rumble_bt_after_input.GetValue(),
        // 0.5.0 exposes one Bluetooth basic mode; it replaces the old "5 ms envelope while
        // idle" checkbox, whose ini key is kept but no longer read.
        .bt_fast = s.hd_rumble_bt_mode.GetValue() == 0,
        .bt_crisp_notes = s.hd_rumble_bt_crisp_notes.GetValue(),
        .bt_mode = std::clamp(s.hd_rumble_bt_mode.GetValue(), 0, 1),
        .usb_no_queue = s.hd_rumble_usb_no_queue.GetValue(),
        .frequency_both_bands = s.hd_rumble_frequency_both_bands.GetValue(),
        .usb_auto_slow = s.hd_rumble_usb_auto_slow.GetValue(),
        .dualsense_enabled = s.dualsense_haptics_enabled.GetValue(),
        .dualsense_volume = std::clamp(s.dualsense_haptics_volume.GetValue(), 0, 100),
        .dualsense_crisp_notes = s.dualsense_crisp_notes.GetValue(),
        .dualsense_hd_haptics = s.dualsense_hd_haptics.GetValue(),
        .dualsense_speaker_volume = std::clamp(s.dualsense_speaker_volume.GetValue(), 0, 100),
        .xbox_enabled = s.xbox_rumble_enabled.GetValue(),
        .xbox_strength = std::clamp(s.xbox_motor_strength.GetValue(), 0, 1000),
        .xbox_direct = s.xbox_rumble_mode.GetValue() == 1,
        .xbox_detail_mix = std::clamp(s.xbox_detail_mix.GetValue(), 0, 100),
        .xbox_hit_accents = std::clamp(s.xbox_hit_accents.GetValue(), 0, 100),
        .xbox_trigger_mix = std::clamp(s.xbox_trigger_mix.GetValue(), 0, 100),
        .gain = std::clamp(s.hd_rumble_gain.GetValue(), 25, 1000),
        .low_boost = std::clamp(s.hd_rumble_low_boost.GetValue(), 0, 100),
    };
}

// Gain, then a power curve that lifts weak vibrations more than strong ones (exponent 1.0 at
// boost 0 down to 0.4 at boost 100). Below a small knee the curve turns quadratic, so the
// near-silent values games send between hits do not become a constant buzz.
float ShapeAmplitude(float amplitude, const UserSettings& u) {
    const float x = (std::min)(amplitude * static_cast<float>(u.gain) / 100.0f, 1.0f);
    if (x <= 0.0f || u.low_boost <= 0) {
        return x;
    }
    const float exponent = 1.0f - 0.006f * static_cast<float>(u.low_boost);
    constexpr float knee = 0.02f;
    if (x < knee) {
        const float r = x / knee;
        return std::pow(knee, exponent) * r * r;
    }
    return std::pow(x, exponent);
}

// Controllers RenHDrum drives, per kind, for the settings dialog
std::array<std::atomic<int>, 3> connected_count{};

// Pads per kind (0 Switch, 1 DualSense, 2 Xbox). Callers only index a kind they know is valid.
constexpr std::size_t ConnectedCountIndex(int kind) {
    return static_cast<std::size_t>(kind);
}
std::atomic<int> last_connected_kind{-1};
// Connected Xbox-style controllers that expose trigger motors
std::atomic<int> xbox_trigger_pads{};
// Connected official (Microsoft) Xbox controllers whose trigger motors SDL does not expose
std::atomic<int> xbox_official_no_triggers{};

} // namespace

// Lock rule: `mutex` is never held while calling an SDL function that takes the SDL joystick lock
// (SDL_SendJoystickEffect, SDL_HDRumbleSet*, SDL_GetJoystickConnectionState, SDL_Rumble*). SDL
// runs the joystick-removed event watcher under that lock, and Shutdown() takes `mutex` from
// there. The joystick pointer is copied out and used after unlocking: SDL3 validates joystick
// handles, so a call on one closed meanwhile only fails.
//
// A DualSense on Bluetooth has no audio endpoint. A thread of its own sends the haptics as HID
// output report 0x32 on a second handle to the pad. That thread never takes `mutex`; it holds
// `bt_dsp_mutex` while it reads the DSP, and the DSP is replaced only under that lock too.
struct RenHDrumPad::Impl {
    bool is_bluetooth{};
    int kind{-1};
    // Rate limit for the refused-packet warning; only Tick writes it, but from two threads
    std::atomic<std::int64_t> next_effect_error_us{0};

    mutable std::mutex mutex;
    SDL_Joystick* joystick{};
    SDL_Gamepad* gamepad{};
    RenHdcEngine* engine{};
    RenHdcDualSenseDsp* dsp{};
    SDL_AudioStream* audio_stream{};
    bool audio_inited{};
    bool audio_opening{};
    int audio_attempts{};
    std::int64_t audio_next_us{};
    UserSettings user{};
    // Switch 2 Mode last handed to the engine; -1 forces the first update
    /// hidapi answers wired or wireless; anything else arrived through the kernel's driver
    /// (Linux hid-nintendo over Bluetooth) and refuses every raw rumble packet, so Vibrate
    /// does not claim those pads and the standard rumble runs instead.
    bool raw_transport{};

    RenHdcXboxRumble* xbox{};
    bool xbox_triggers{};
    bool official_no_triggers{};
    // Direct mode: the latest shaped bands of each side, and the grip values last sent
    std::array<float, 2> direct_low{};
    std::array<float, 2> direct_high{};
    std::array<std::uint16_t, 2> direct_sent{};
    // Direct mode accents: the level to settle on once the kick ends
    std::array<std::uint16_t, 2> direct_target{};
    std::int64_t kick_until_us{};
    /// Direct-mode bookkeeping: the sent state is only recorded once SDL accepts a write
    /// (pending forces the next sample to retry), a steady nonzero level is renewed before
    /// SDL's 2000 ms expiry, and send_mutex serializes this pad's rumble writes against
    /// Tick's settle so a stale level cannot land over a newer one.
    bool direct_pending{};
    bool trigger_pending{};
    std::int64_t last_direct_ok_us{};
    std::int64_t last_trigger_ok_us{};
    std::uint64_t direct_generation{};
    std::mutex send_mutex;
    // Direct mode triggers: the fine vibration of each side, and what was last sent
    std::array<std::uint16_t, 2> trigger_target{};
    std::array<std::uint16_t, 2> trigger_sent{};

    mutable std::mutex bt_dsp_mutex;
    SDL_hid_device* bt_hid{};
    std::thread bt_thread;
    std::atomic<bool> bt_running{};
    // HD Haptics (RenHDrum 0.5.0): one trigger-kick handle per side, the latest raw sample and
    // fresh flag for it, and the speaker state. `tri_mutex` guards the trigger step; lock order is
    // `tri_mutex`, then `mutex`.
    std::array<RenHdcTriTrigger*, 2> tri{};
    std::mutex tri_mutex;
    std::array<std::array<float, 4>, 2> tri_sample{};
    std::array<bool, 2> tri_fresh{};
    bool speaker_on{};
    int speaker_volume_applied{};
    bool speaker_logged{};
    bool speaker_off_logged{};
    // The combined Bluetooth report 0x36 carries the triggers; the sender thread reads both
    // sides' latest effects, written under `bt_dsp_mutex`.
    std::array<std::array<Uint8, 11>, 2> bt_effect{};
    std::array<bool, 2> bt_effect_set{};
    std::atomic<bool> bt_report36{};
    // Set by Release() when the pad leaves, so Shutdown() can turn the extras off over HID
    bool restore_speaker{};
    bool restore_trigger{};

    /// Engine settings from the user's choices (Ryuhdr 0.2.6's recommended mode by default).
    /// Caller holds the mutex.
    bool ResolveEngineSettings(RenHdcSettings& settings) const {
        RenHdcConfig config{};
        config.enable_hd_rumble = user.hd_enabled;
        config.hd_rumble_usb_interval_ms = user.usb_interval_ms;
        config.hd_rumble_bluetooth_level = user.bluetooth_level;
        config.hd_rumble_usb_auto_slow = user.usb_auto_slow;
        config.hd_rumble_experimental_usb_no_queue = user.usb_no_queue;
        config.hd_rumble_experimental_envelope_5ms = user.envelope_5ms;
        config.hd_rumble_experimental_frequency_in_packet = user.frequency_in_packet;
        config.hd_rumble_experimental_frequency_both_bands = user.frequency_both_bands;
        config.hd_rumble_experimental_crisp_notes = user.crisp_notes;
        config.hd_rumble_experimental_bt_after_input = user.bt_after_input;
        config.hd_rumble_experimental_bt_fast = user.bt_fast;
        config.hd_rumble_experimental_bt_crisp_notes = user.bt_crisp_notes;
        // HD Haptics is on by default in 0.5.0; this is the single switch the core knows. The
        // speaker and trigger wiring below is what actually makes it happen on the pad.
        config.dualsense_hd_haptics_off = !user.dualsense_hd_haptics;
        const auto& api = Api();
        const auto resolve = api.settings_resolve_pure ? api.settings_resolve_pure
                                                       : api.settings_resolve;
        return resolve(&config, &settings) == RENHDRUM_OK;
    }

    /// Switches SDL's HD Rumble transport modes to match `settings`, then clears the modes SDL
    /// refused, so the engine never paces for writes SDL does not gate. Called without the mutex.
    void ApplyTransportModes(SDL_Joystick* js, RenHdcSettings& settings) const {
#ifdef RENHDRUM_SDL_HDRUMBLE
        SDL_HDRumbleSetNoQueue(js, 0, 0, 0);
        SDL_HDRumbleSetNoQueueBluetooth(js, 0, 0);
        SDL_HDRumbleSetBluetoothAfterInput(js, 0);
        if (is_bluetooth) {
            // B3 needs B1 on; SDL never runs both on one pad, B3 wins
            settings.bt_fast = settings.enabled && settings.bt_fast && settings.bt_after_input &&
                               SDL_HDRumbleSetNoQueueBluetooth(js, 1, NoQueueEnvelopeGapMs) == 1;
            settings.bt_after_input = settings.enabled && settings.bt_after_input &&
                                      !settings.bt_fast &&
                                      SDL_HDRumbleSetBluetoothAfterInput(js, 1) == 1;
            settings.usb_no_queue = false;
        } else {
            const int gap_ms =
                settings.envelope_5ms ? NoQueueEnvelopeGapMs : settings.usb_interval_ms;
            settings.usb_no_queue =
                settings.enabled && settings.usb_no_queue &&
                SDL_HDRumbleSetNoQueue(js, 1, gap_ms, settings.usb_pause_ms) == 1;
            settings.bt_after_input = false;
            settings.bt_fast = false;
        }
#else
        // Stock SDL queues every write; the engine paces by its clock.
        (void)js;
        settings.usb_no_queue = false;
        settings.bt_after_input = false;
        settings.bt_fast = false;
#endif
    }

    /// Resolves the settings, switches the SDL modes and hands the result to the engine.
    /// Called without the mutex.
    void ConfigureEngine() {
        RenHdcSettings settings{};
        SDL_Joystick* js = nullptr;
        {
            std::scoped_lock lock{mutex};
            if (!engine || !joystick || !ResolveEngineSettings(settings)) {
                return;
            }
            js = joystick;
        }
        ApplyTransportModes(js, settings);
        {
            std::scoped_lock lock{mutex};
            if (!engine) {
                return;
            }
            Api().engine_apply_settings(engine, &settings);
        }
        if (is_bluetooth) {
            LOG_INFO(Input, "RenHDrum HD Rumble Bluetooth: level {}, {}{}", settings.bluetooth_level,
                     settings.bt_fast          ? "5 ms envelope while SDL is idle"
                     : settings.bt_after_input ? "written right after an input report"
                                               : "clock based writes",
                     settings.enabled && settings.bt_crisp_notes ? ", crisp notes" : "");
        } else {
            const bool envelope = settings.usb_no_queue && settings.envelope_5ms;
            LOG_INFO(Input, "RenHDrum HD Rumble wired: {} ms, {}{}{}", settings.usb_interval_ms,
                     settings.usb_no_queue ? "no-queue writes" : "queued writes",
                     envelope ? ", 5 ms envelope" : "",
                     envelope && settings.crisp_notes ? ", crisp notes" : "");
        }
    }

    struct SettingsChange {
        bool reconfigure_engine{};
        bool stop_engine{};
        bool sync_hd_haptics{};
    };

    /// Picks up changes made in the HD Rumble dialog. Caller holds the mutex; the returned SDL
    /// work is done after unlocking.
    SettingsChange RefreshUserSettings() {
        const UserSettings now = ReadUserSettings();
        if (now == user) {
            return {};
        }
        const UserSettings old = user;
        user = now;
        if (dsp && (old.dualsense_volume != now.dualsense_volume ||
                    old.dualsense_crisp_notes != now.dualsense_crisp_notes)) {
            // The audio callback runs with the stream locked, so it never sees a freed DSP. The
            // audio stream lock is not the joystick lock, and the callback never takes `mutex`.
            if (audio_stream) {
                SDL_LockAudioStream(audio_stream);
            }
            {
                std::scoped_lock bt_lock{bt_dsp_mutex};
                const auto& api = Api();
                api.dsp_destroy(dsp);
                dsp = api.dsp_create(now.dualsense_volume, now.dualsense_crisp_notes);
                // The new DSP has no speaker applied yet, even when the volume did not
                // change; forget the applied level so SyncHdHaptics sets it again. The
                // separate speaker_on state stays: it says the physical speaker is on.
                speaker_volume_applied = -1;
            }
            if (audio_stream) {
                SDL_UnlockAudioStream(audio_stream);
            }
        }
        return {
            .reconfigure_engine = engine != nullptr,
            .stop_engine = engine != nullptr && old.hd_enabled && !now.hd_enabled,
            // The volume and crisp-notes rows recreate the DSP, which forgets the speaker, so a
            // recreate also re-syncs. Everything else just flips what SyncHdHaptics applies.
            .sync_hd_haptics = dsp != nullptr &&
                               (old.hd_enabled != now.hd_enabled ||
                                old.dualsense_enabled != now.dualsense_enabled ||
                                old.dualsense_hd_haptics != now.dualsense_hd_haptics ||
                                old.dualsense_speaker_volume != now.dualsense_speaker_volume ||
                                old.dualsense_volume != now.dualsense_volume ||
                                old.dualsense_crisp_notes != now.dualsense_crisp_notes),
        };
    }

    /// Brings HD Haptics in line with the settings: on = the speaker (wired, once its audio
    /// stream is open) and a trigger-kick handle per side, plus report 0x36 on Bluetooth; off =
    /// both triggers off and the speaker released. Called without the mutex.
    void SyncHdHaptics() {
        const auto& api = Api();
        SDL_Joystick* js = nullptr;
        bool open_speaker = false;
        bool close_speaker = false;
        bool off_triggers = false;
        bool log_trigger = false;
        {
            std::scoped_lock lock{mutex};
            if (!dsp || !joystick) {
                return;
            }
            js = joystick;
            const bool want = api.HasHdHaptics() && user.hd_enabled && user.dualsense_enabled &&
                              user.dualsense_hd_haptics;
            const bool bt36 = want && is_bluetooth && api.dsp_build_bt36_report != nullptr;
            bt_report36.store(bt36);
            if (want && gamepad) {
                for (auto& handle : tri) {
                    if (!handle) {
                        handle = api.tri_create();
                        log_trigger |= handle && !std::exchange(speaker_logged, true);
                    }
                }
            } else {
                for (auto& handle : tri) {
                    if (handle) {
                        api.tri_destroy(handle);
                        handle = nullptr;
                        off_triggers = true;
                    }
                }
            }
            if (!is_bluetooth) {
                if (want && audio_stream && !speaker_on) {
                    api.dsp_set_speaker(dsp, user.dualsense_speaker_volume);
                    speaker_on = true;
                    open_speaker = true;
                } else if (want && speaker_on &&
                           speaker_volume_applied != user.dualsense_speaker_volume) {
                    api.dsp_set_speaker(dsp, user.dualsense_speaker_volume);
                } else if (!want && speaker_on) {
                    api.dsp_set_speaker(dsp, 0);
                    speaker_on = false;
                    close_speaker = true;
                }
                speaker_volume_applied = want ? user.dualsense_speaker_volume : 0;
            } else if (api.dsp_set_speaker) {
                std::scoped_lock bt_lock{bt_dsp_mutex};
                api.dsp_set_speaker(dsp, bt36 ? user.dualsense_speaker_volume : 0);
                speaker_volume_applied = bt36 ? user.dualsense_speaker_volume : 0;
            }
        }
        if (open_speaker) {
            const auto block = HdSpeakerBlock(true);
            SDL_SendJoystickEffect(js, block.data(), static_cast<int>(block.size()));
            LOG_INFO(Input, "RenHDrum: DualSense speaker on ({}%)", speaker_volume_applied);
        }
        if (log_trigger) {
            LOG_INFO(Input, "RenHDrum: DualSense trigger kick on");
        }
        if (off_triggers) {
            {
                std::scoped_lock bt_lock{bt_dsp_mutex};
                bt_effect_set = {};
            }
            for (const bool is_right : {false, true}) {
                const auto block = HdTriggerBlock(is_right, nullptr);
                SDL_SendJoystickEffect(js, block.data(), static_cast<int>(block.size()));
            }
        }
        if (close_speaker) {
            const auto block = HdSpeakerBlock(false);
            SDL_SendJoystickEffect(js, block.data(), static_cast<int>(block.size()));
        }
    }

    /// Caller holds `bt_dsp_mutex`. The 47-byte trigger state for report 0x36 (both sides' latest
    /// effects), or false while no trigger effect exists yet.
    bool BuildBtTriggerState(Ds5Block& out) const {
        if (!bt_effect_set[0] && !bt_effect_set[1]) {
            return false;
        }
        out = {};
        for (const bool is_right : {false, true}) {
            const std::size_t side = is_right ? 1 : 0;
            if (!bt_effect_set[side]) {
                continue;
            }
            const auto one = HdTriggerBlock(is_right, &bt_effect[side]);
            for (std::size_t i = 0; i < out.size(); ++i) {
                out[i] |= one[i];
            }
        }
        return true;
    }

    /// One step of the trigger kick for one side; Tick calls it every ~1 ms. Vibrate only stores
    /// the newest raw sample and the fresh flag. Lock order: `tri_mutex`, then `mutex`. Called
    /// without `mutex`.
    void TriggerStep(std::int64_t now_us, bool is_right) {
        std::scoped_lock tri_lock{tri_mutex};
        const std::size_t side = is_right ? 1 : 0;
        SDL_Joystick* js = nullptr;
        SDL_Gamepad* gp = nullptr;
        {
            std::scoped_lock lock{mutex};
            if (!tri[side] || !joystick || !gamepad) {
                return;
            }
            js = joystick;
            gp = gamepad;
        }
        const int axis =
            std::clamp<int>(SDL_GetGamepadAxis(gp, is_right ? SDL_GAMEPAD_AXIS_RIGHT_TRIGGER
                                                            : SDL_GAMEPAD_AXIS_LEFT_TRIGGER),
                            0, 32767);
        const auto depth = static_cast<Uint8>(axis * 255 / 32767);
        // Another action button (or the other trigger) held means the rumble belongs to that
        // action, so this trigger must not follow it. START/BACK/GUIDE/TOUCHPAD/MISC/paddles do
        // not count.
        static constexpr std::array<SDL_GamepadButton, 12> action_buttons{
            SDL_GAMEPAD_BUTTON_SOUTH,         SDL_GAMEPAD_BUTTON_EAST,
            SDL_GAMEPAD_BUTTON_WEST,          SDL_GAMEPAD_BUTTON_NORTH,
            SDL_GAMEPAD_BUTTON_LEFT_SHOULDER, SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER,
            SDL_GAMEPAD_BUTTON_LEFT_STICK,    SDL_GAMEPAD_BUTTON_RIGHT_STICK,
            SDL_GAMEPAD_BUTTON_DPAD_UP,       SDL_GAMEPAD_BUTTON_DPAD_DOWN,
            SDL_GAMEPAD_BUTTON_DPAD_LEFT,     SDL_GAMEPAD_BUTTON_DPAD_RIGHT};
        bool other_action =
            SDL_GetGamepadAxis(gp, is_right ? SDL_GAMEPAD_AXIS_LEFT_TRIGGER
                                            : SDL_GAMEPAD_AXIS_RIGHT_TRIGGER) *
                255 / 32767 >
            12;
        for (std::size_t i = 0; !other_action && i < action_buttons.size(); ++i) {
            other_action = SDL_GetGamepadButton(gp, action_buttons[i]);
        }
        std::array<Uint8, 11> effect{};
        int changed = 0;
        {
            std::scoped_lock lock{mutex};
            if (!tri[side]) {
                return;
            }
            const bool fresh = std::exchange(tri_fresh[side], false);
            const auto& s = tri_sample[side];
            const auto now_ms = static_cast<std::uint32_t>(now_us / 1000);
            if (other_action && Api().tri_note_action) {
                Api().tri_note_action(tri[side], now_ms);
            }
            changed = Api().tri_update(tri[side], now_ms, depth, s[0], s[1], s[2], s[3], fresh,
                                       effect.data());
        }
        if (changed == 1) {
            if (bt_report36.load()) {
                // Bluetooth: the sender thread carries it in the next report 0x36
                std::scoped_lock bt_lock{bt_dsp_mutex};
                bt_effect[side] = effect;
                bt_effect_set[side] = true;
            } else {
                const auto block = HdTriggerBlock(is_right, &effect);
                SDL_SendJoystickEffect(js, block.data(), static_cast<int>(block.size()));
            }
        }
    }

    /// The DualSense exposes its voice coils as a 4-channel USB audio device that Windows may
    /// list a moment after the HID device, hence the retries. Called without the mutex.
    void OpenDualSenseAudio() {
        SDL_Joystick* js = nullptr;
        SDL_Gamepad* gp = nullptr;
        {
            std::scoped_lock lock{mutex};
            if (!dsp || !joystick || audio_stream || audio_opening) {
                return;
            }
            if (is_bluetooth) {
                // No audio endpoint on Bluetooth: one try at the HID haptics, no retries
                audio_attempts = DualSenseAudioMaxAttempts;
            } else {
                audio_opening = true;
                ++audio_attempts;
            }
            js = joystick;
            gp = gamepad;
        }
        if (is_bluetooth) {
            StartDualSenseBluetooth(js, gp);
            return;
        }
        // SDL counts subsystem references: every pad with a stream holds one, so one pad going
        // away does not stop another's audio.
        SDL_AudioStream* stream = nullptr;
        const bool inited = SDL_InitSubSystem(SDL_INIT_AUDIO);
        if (inited) {
            stream = OpenDualSenseAudioStream(js, audio_attempts == 1);
        } else {
            LOG_WARNING(Input, "RenHDrum: SDL audio init failed: {}", SDL_GetError());
        }
        bool opened = false;
        {
            std::scoped_lock lock{mutex};
            audio_opening = false;
            if (stream && joystick && dsp && !audio_stream) {
                audio_stream = stream;
                audio_inited = true;
                opened = true;
                // Start only once published: from here on, the DSP is replaced or freed only
                // under SDL_LockAudioStream or after SDL_DestroyAudioStream. The callback never
                // takes `mutex`.
                if (!SDL_ResumeAudioStreamDevice(stream)) {
                    // Not playing: unpublish so vibration falls back to standard rumble and
                    // the remaining retries can still open the device. The callback never ran.
                    LOG_WARNING(Input, "RenHDrum: DualSense audio did not start: {}",
                                SDL_GetError());
                    audio_stream = nullptr;
                    audio_inited = false;
                    opened = false;
                }
            }
        }
        if (!opened) {
            // No device yet, or shut down while opening
            if (stream) {
                SDL_DestroyAudioStream(stream);
            }
            if (inited) {
                SDL_QuitSubSystem(SDL_INIT_AUDIO);
            }
        }
        if (opened) {
            // Stop SDL's HID rumble once so the pad plays the audio haptics
            if (gp) {
                SDL_RumbleGamepad(gp, 0, 0, 0);
            } else {
                SDL_RumbleJoystick(js, 0, 0, 0);
            }
            // The speaker can only open now that its audio stream exists
            SyncHdHaptics();
        }
    }

    /// Opens a second HID handle on a Bluetooth DualSense and starts the report 0x32 sender.
    /// Called without the mutex. Without the 0.2.1 function or a HID path, the standard rumble
    /// stays.
    void StartDualSenseBluetooth(SDL_Joystick* js, SDL_Gamepad* gp) {
        if (!Api().dsp_build_bt_report) {
            LOG_INFO(Input, "RenHDrum: DualSense on Bluetooth needs RenHDrum 0.2.1 or newer, "
                            "using the standard rumble");
            return;
        }
        const char* path = SDL_GetJoystickPath(js);
        if (!path || !*path) {
            LOG_INFO(Input, "RenHDrum: DualSense on Bluetooth has no HID path, using the "
                            "standard rumble");
            return;
        }
        const std::string hid_path{path};
        SDL_hid_device* hid = SDL_hid_open_path(hid_path.c_str());
        if (!hid) {
            LOG_WARNING(Input,
                        "RenHDrum: could not open the DualSense HID path: {}, using the "
                        "standard rumble",
                        SDL_GetError());
            return;
        }
        {
            std::scoped_lock lock{mutex};
            if (!joystick || !dsp || bt_hid) {
                SDL_hid_close(hid);
                return;
            }
            bt_hid = hid;
            bt_running.store(true);
            bt_thread = std::thread([this] { DualSenseBluetoothLoop(); });
        }
        // Stop SDL's HID rumble once so the pad plays the haptics
        if (gp) {
            SDL_RumbleGamepad(gp, 0, 0, 0);
        } else {
            SDL_RumbleJoystick(js, 0, 0, 0);
        }
        LOG_INFO(Input, "RenHDrum: DualSense haptics over Bluetooth (report 0x32)");
    }

    /// Runs on `bt_thread`: one report every RENHDRUM_DS5_BT_PERIOD_NS while vibrating and for
    /// 2 s after, then quiet on the radio. Stops after about 0.5 s of failed writes, which hands
    /// the vibration back to the standard rumble.
    void DualSenseBluetoothLoop() {
        Common::SetCurrentThreadName("RenHDrumDS5BT");
        constexpr Uint64 KeepSendingNs = 2'000'000'000ULL;
        constexpr Uint64 ResyncNs = 100'000'000ULL;
        constexpr int MaxFailsInRow = 50;
        const auto& api = Api();
        std::array<std::uint8_t, RENHDRUM_DS5_BT_REPORT_SIZE> report{};
        std::array<std::uint8_t, RENHDRUM_DS5_BT36_REPORT_SIZE> report36{};
        Uint64 next = SDL_GetTicksNS();
        Uint64 last_signal = 0;
        int fails_in_row = 0;
        while (bt_running.load()) {
            next += RENHDRUM_DS5_BT_PERIOD_NS;
            // HD Haptics on: the combined report 0x36 (rumble + triggers + speaker), which the
            // core stops by itself after 2 s of silence. Off: the plain 0x32 as before.
            const bool combined = bt_report36.load();
            int built = -1;
            {
                std::scoped_lock lock{bt_dsp_mutex};
                if (dsp) {
                    if (combined) {
                        Ds5Block state{};
                        const bool have_state = BuildBtTriggerState(state);
                        built = api.dsp_build_bt36_report(dsp, have_state ? state.data() : nullptr,
                                                          0, report36.data());
                    } else {
                        built = api.dsp_build_bt_report(dsp, RENHDRUM_DS5_BT_BUFFER_LENGTH_DEFAULT,
                                                        report.data());
                    }
                }
            }
            const Uint64 now = SDL_GetTicksNS();
            if (built == 1) {
                last_signal = now;
            }
            const bool send = combined ? built == 1
                                       : built >= 0 && last_signal != 0 &&
                                             now - last_signal < KeepSendingNs;
            if (send) {
                const int written =
                    combined ? SDL_hid_write(bt_hid, report36.data(), report36.size())
                             : SDL_hid_write(bt_hid, report.data(), report.size());
                if (written < 0) {
                    ++fails_in_row;
                } else {
                    fails_in_row = 0;
                }
            }
            if (fails_in_row >= MaxFailsInRow) {
                LOG_WARNING(Input,
                            "RenHDrum: DualSense Bluetooth writes failing ({}), using the "
                            "standard rumble",
                            SDL_GetError());
                break;
            }
            if (next > now) {
                SDL_DelayPrecise(next - now);
            } else if (now - next > ResyncNs) {
                next = now; // stalled: resync instead of bursting
            }
        }
        bt_running.store(false);
    }

    /// Finds the DualSense audio device and opens a stream on it, paused until it is published
    /// in `audio_stream`, or nullptr. Called without the mutex, with the SDL audio subsystem
    /// initialized.
    SDL_AudioStream* OpenDualSenseAudioStream(SDL_Joystick* js, bool verbose) {
        int count = 0;
        SDL_AudioDeviceID* devices = SDL_GetAudioPlaybackDevices(&count);
        SDL_AudioDeviceID matched = 0;
        for (int i = 0; devices && i < count; ++i) {
            const char* name = SDL_GetAudioDeviceName(devices[i]);
            SDL_AudioSpec spec{};
            const int channels =
                SDL_GetAudioDeviceFormat(devices[i], &spec, nullptr) ? spec.channels : 0;
            // Names differ per platform: WASAPI and PipeWire say "DualSense" or "Wireless
            // Controller", CoreAudio spells out the whole product name.
            const bool looks_like_dualsense =
                name && (std::strstr(name, "DualSense") != nullptr ||
                         std::strstr(name, "DualSense Edge") != nullptr ||
                         std::strstr(name, "Wireless Controller") != nullptr);
            if (!looks_like_dualsense) {
                continue;
            }
            if (channels >= 4) {
                matched = devices[i];
                break;
            }
            // The voice coils live on channels 3-4. A two-channel endpoint means the OS exposed
            // the pad as plain stereo (WASAPI shared mode does, and so does Bluetooth everywhere).
            LOG_INFO(Input,
                     "RenHDrum: \"{}\" exposes {} channel(s); haptics need 4, wired USB only",
                     name, channels);
        }
        if (matched == 0 && verbose) {
            // Once per attempt sequence, so a report from a platform we cannot test here says what
            // the audio server actually offered.
            LOG_INFO(Input, "RenHDrum: no 4-channel DualSense device yet, {} playback device(s):",
                     count);
            for (int i = 0; devices && i < count; ++i) {
                SDL_AudioSpec spec{};
                const int channels =
                    SDL_GetAudioDeviceFormat(devices[i], &spec, nullptr) ? spec.channels : 0;
                LOG_INFO(Input, "RenHDrum:   [{}] \"{}\" {} channel(s)", devices[i],
                         SDL_GetAudioDeviceName(devices[i]), channels);
            }
        }
        SDL_free(devices);
        if (matched == 0) {
            LOG_DEBUG(Input, "RenHDrum: no DualSense audio device yet");
            return nullptr;
        }

        const SDL_AudioSpec spec{
            .format = SDL_AUDIO_F32,
            .channels = 4,
            .freq = 48000,
        };
        // Four channels, because the DualSense voice coils listen on 3-4 and the pad exposes them
        // only through its raw UAC endpoint. Anything that swallows the stream between us and that
        // endpoint can flatten it to stereo and the coils fall silent: measured with a PipeWire
        // filter-chain sink (JamesDSP) as the default sink, a four-channel feed arrived as two
        // channels and the game audio took the pad's ports instead. Nothing on this side can see
        // or fix that -- it is a routing decision above us -- so the device name and the channel
        // counts below are logged to make it diagnosable in one run.
        // Give the stream a stable name so it is identifiable in an audio graph and can be pinned
        // by a server-side rule; every stream would otherwise be called plain "eglantine".
        SDL_SetHint(SDL_HINT_AUDIO_DEVICE_STREAM_NAME, "RenHDrum haptics");
        SDL_AudioStream* stream = SDL_OpenAudioDeviceStream(matched, &spec, &AudioCallback, this);
        SDL_ResetHint(SDL_HINT_AUDIO_DEVICE_STREAM_NAME);
        if (!stream) {
            LOG_WARNING(Input, "RenHDrum: could not open the DualSense audio device: {}",
                        SDL_GetError());
            return nullptr;
        }
        // No channel-map call here: SDL 3.4.8's map only reorders or mutes channels (`{1,0}` swaps
        // stereo) and has no way to name speaker positions -- that enum is a later SDL. So what the
        // four channels are has to come from the device itself; this log line is how we tell
        // whether the server gave us a four-channel endpoint or a downmix.
        SDL_AudioSpec src_spec{};
        SDL_AudioSpec dst_spec{};
        if (SDL_GetAudioStreamFormat(stream, &src_spec, &dst_spec)) {
            LOG_INFO(Input,
                     "RenHDrum: DualSense haptics stream: feed {} channel(s) at {} Hz, device takes "
                     "{} at {} Hz (\"{}\")",
                     src_spec.channels, src_spec.freq, dst_spec.channels, dst_spec.freq,
                     SDL_GetAudioDeviceName(matched));
        }
        LOG_INFO(Input, "RenHDrum: DualSense haptics on \"{}\"", SDL_GetAudioDeviceName(matched));
        return stream;
    }

    static void SDLCALL AudioCallback(void* userdata, SDL_AudioStream* stream,
                                      int additional_amount, int /*total_amount*/) {
        auto* const self = static_cast<Impl*>(userdata);
        if (additional_amount <= 0 || !self || !self->dsp) {
            return;
        }
        constexpr int frame_bytes = 4 * static_cast<int>(sizeof(float));
        constexpr int chunk_frames = 256;
        std::array<float, chunk_frames * 4> chunk{};
        int frames = additional_amount / frame_bytes;
        while (frames > 0) {
            const int count = (std::min)(frames, chunk_frames);
            if (Api().dsp_fill_f32(self->dsp, chunk.data(), static_cast<size_t>(count)) !=
                RENHDRUM_OK) {
                chunk.fill(0.0f); // never send garbage to the actuators
            }
            SDL_PutAudioStreamData(stream, chunk.data(), count * frame_bytes);
            frames -= count;
        }
    }

    /// Frees the engine, the DSP and the audio stream. Caller holds the mutex. Returns whether
    /// an engine was running, so the caller sends the neutral packet after unlocking.
    bool had_xbox{};

    bool Release() {
        const auto& api = Api();
        // The sender never takes `mutex`, so joining here cannot deadlock
        bt_running.store(false);
        if (bt_thread.joinable()) {
            bt_thread.join();
        }
        if (bt_hid) {
            SDL_hid_close(bt_hid);
            bt_hid = nullptr;
        }
        if (audio_stream) {
            // Waits for a running audio callback; the callback never takes `mutex`
            SDL_DestroyAudioStream(audio_stream);
            audio_stream = nullptr;
        }
        if (audio_inited) {
            SDL_QuitSubSystem(SDL_INIT_AUDIO);
            audio_inited = false;
        }
        if (xbox) {
            api.xbox_destroy(xbox);
            xbox = nullptr;
            had_xbox = true;
        }
        const bool had_engine = engine != nullptr;
        if (engine) {
            api.engine_destroy(engine);
            engine = nullptr;
        }
        // HD Haptics: the handles die with the DSP; the pad-side state does not, so remember
        // what Shutdown() has to turn off over HID while the joystick is still there.
        if (speaker_on && dsp && api.dsp_set_speaker) {
            api.dsp_set_speaker(dsp, 0);
        }
        restore_speaker = std::exchange(speaker_on, false);
        for (auto& handle : tri) {
            if (handle) {
                api.tri_destroy(handle);
                handle = nullptr;
                restore_trigger = true;
            }
        }
        bt_effect_set = {};
        if (dsp) {
            api.dsp_destroy(dsp);
            dsp = nullptr;
        }
        return had_engine;
    }
};

static void SendNeutralPacket(SDL_Joystick* joystick) {
    SDL_SendJoystickEffect(joystick, NeutralPacket.data(), static_cast<int>(NeutralPacket.size()));
}

bool IsRenHDrumBuilt() {
    return true;
}

bool IsRenHDrumLoaded() {
    return Api().loaded;
}

std::vector<RenHDrumOption> GetRenHDrumOptions() {
    const auto& api = Api();
    if (api.loaded && api.options) {
        int32_t count = 0;
        int32_t stride = 0;
        const RenHdcOption* table = api.options(&count, &stride);
        // The core hands out the row stride so a host built against an older header can never
        // misstep through a newer, larger row.
        if (table != nullptr && count > 0 && stride >= static_cast<int32_t>(sizeof(RenHdcOption))) {
            std::vector<RenHDrumOption> out;
            out.reserve(static_cast<std::size_t>(count));
            for (int32_t i = 0; i < count; ++i) {
                const auto& row = *reinterpret_cast<const RenHdcOption*>(
                    reinterpret_cast<const char*>(table) +
                    static_cast<std::size_t>(i) * static_cast<std::size_t>(stride));
                RenHDrumOption option;
                option.key = row.key != nullptr ? row.key : "";
                option.label = row.label != nullptr ? row.label : "";
                option.help = row.help != nullptr ? row.help : "";
                if (row.choices != nullptr && *row.choices != '\0') {
                    std::string choices{row.choices};
                    std::size_t start = 0;
                    while (true) {
                        const std::size_t bar = choices.find('|', start);
                        option.choices.push_back(choices.substr(start, bar - start));
                        if (bar == std::string::npos) {
                            break;
                        }
                        start = bar + 1;
                    }
                }
                option.kind = row.kind;
                option.min = row.min;
                option.max = row.max;
                option.step = row.step;
                option.def = row.def;
                option.devices = static_cast<unsigned>(row.devices);
                option.tier = row.tier;
                option.link = row.link;
                option.requires_row = row.requires_row;
                out.push_back(std::move(option));
            }
            return out;
        }
    }
    return FallbackRenHDrumOptions();
}

std::shared_ptr<RenHDrumPad> RenHDrumPad::Create(SDL_Joystick* joystick, SDL_Gamepad* gamepad) {
    if (!joystick || !IsRenHDrumLoaded()) {
        return nullptr;
    }
    const Uint16 vid = SDL_GetJoystickVendor(joystick);
    const Uint16 pid = SDL_GetJoystickProduct(joystick);
    const bool switch_pad = IsSwitchHdController(vid, pid);
    const auto& api_check = Api();
    const bool xbox_pad = !switch_pad && !IsDualSense(vid, pid) && api_check.HasXbox() &&
                          IsXboxStyle(joystick, gamepad);
    if (!switch_pad && !xbox_pad && !IsDualSense(vid, pid)) {
        LOG_INFO(Input, "RenHDrum: {:04X}:{:04X} \"{}\" is not a controller RenHDrum drives",
                 vid, pid, SDL_GetJoystickName(joystick) ? SDL_GetJoystickName(joystick) : "");
        return nullptr;
    }

    // Not shared yet: no other thread can reach `state` until it is returned.
    const auto& api = Api();
    auto state = std::make_unique<Impl>();
    state->joystick = joystick;
    state->gamepad = gamepad;
    state->is_bluetooth =
        SDL_GetJoystickConnectionState(joystick) == SDL_JOYSTICK_CONNECTION_WIRELESS;
    state->user = ReadUserSettings();
    if (xbox_pad) {
        // Unknown connection: the safer Bluetooth pacing (20 ms ticks)
        const bool wired =
            SDL_GetJoystickConnectionState(joystick) == SDL_JOYSTICK_CONNECTION_WIRED;
        state->xbox = api.xbox_create(!wired);
        if (!state->xbox) {
            return nullptr;
        }
        state->xbox_triggers = SDL_GetBooleanProperty(
            SDL_GetJoystickProperties(joystick), SDL_PROP_JOYSTICK_CAP_TRIGGER_RUMBLE_BOOLEAN,
            false);
        const char* gamepad_name = SDL_GetGamepadName(gamepad);
        LOG_INFO(Input,
                 "RenHDrum: Xbox rumble for {:04X}:{:04X} \"{}\" (wired: {}, trigger motors: {})",
                 vid, pid, gamepad_name ? gamepad_name : "", wired, state->xbox_triggers);
    } else if (switch_pad) {
        state->engine = api.engine_create(state->is_bluetooth, pid == 0x2009);
        if (!state->engine) {
            return nullptr;
        }
        LOG_INFO(Input, "RenHDrum: HD Rumble for {:04X}:{:04X} (Bluetooth: {})", vid, pid,
                 state->is_bluetooth);
        // Wired and wireless are the only answers SDL can give when it owns the pad through
        // hidapi. Anything else means the pad came in through evdev -- on Linux that is the
        // kernel's hid-nintendo driver -- and evdev answers every raw rumble packet with
        // SDL_Unsupported, so the engine's packets never leave the host.
        const SDL_JoystickConnectionState connection = SDL_GetJoystickConnectionState(joystick);
        state->raw_transport = connection == SDL_JOYSTICK_CONNECTION_WIRED ||
                               connection == SDL_JOYSTICK_CONNECTION_WIRELESS;
        if (!state->raw_transport) {
            LOG_WARNING(Input,
                        "RenHDrum: SDL reports neither wired nor wireless for this pad, so it "
                        "likely arrived through the kernel's driver and will refuse raw rumble "
                        "packets: HD Rumble stays off for it and the standard rumble runs");
        }
        state->ConfigureEngine();
    } else {
        state->dsp = api.dsp_create(state->user.dualsense_volume, state->user.dualsense_crisp_notes);
        if (!state->dsp) {
            return nullptr;
        }
        LOG_INFO(Input, "RenHDrum: DualSense haptics for {:04X}:{:04X}", vid, pid);
        // HD Haptics: the trigger kick (and report 0x36 on Bluetooth) now; the speaker once its
        // audio stream is open
        state->SyncHdHaptics();
        state->OpenDualSenseAudio();
    }
    state->kind = xbox_pad ? 2 : switch_pad ? 0 : 1;
    ++connected_count[ConnectedCountIndex(state->kind)];
    if (state->xbox_triggers) {
        ++xbox_trigger_pads;
    }
    state->official_no_triggers = xbox_pad && vid == MicrosoftVendorId && !state->xbox_triggers;
    if (state->official_no_triggers) {
        ++xbox_official_no_triggers;
    }
    last_connected_kind = state->kind;
    return std::shared_ptr<RenHDrumPad>(new RenHDrumPad(std::move(state)));
}

int GetRenHDrumConnectedKind() {
    const int last = last_connected_kind.load();
    if (last >= 0 && connected_count[ConnectedCountIndex(last)].load() > 0) {
        return last;
    }
    for (int kind = 0; kind < static_cast<int>(connected_count.size()); ++kind) {
        if (connected_count[ConnectedCountIndex(kind)].load() > 0) {
            return kind;
        }
    }
    return -1;
}

int GetRenHDrumXboxTriggers() {
    if (connected_count[2].load() == 0) {
        return -1;
    }
    return xbox_trigger_pads.load() > 0 ? 1 : 0;
}

bool GetRenHDrumXboxNeedsRawInput() {
#ifdef _WIN32
    return xbox_official_no_triggers.load() > 0;
#else
    return false;
#endif
}

RenHDrumPad::RenHDrumPad(std::unique_ptr<Impl> impl_) : impl{std::move(impl_)} {}

RenHDrumPad::~RenHDrumPad() {
    Shutdown();
}

void RenHDrumPad::Shutdown() {
    SDL_Joystick* joystick = nullptr;
    bool had_engine = false;
    bool had_xbox = false;
    bool triggers = false;
    {
        std::scoped_lock lock{impl->mutex};
        if (!impl->joystick) {
            return;
        }
        joystick = impl->joystick;
        had_engine = impl->Release();
        had_xbox = impl->had_xbox;
        if (impl->kind >= 0) {
            --connected_count[ConnectedCountIndex(impl->kind)];
            if (impl->xbox_triggers) {
                --xbox_trigger_pads;
            }
            if (impl->official_no_triggers) {
                --xbox_official_no_triggers;
            }
            impl->kind = -1;
        }
        triggers = impl->xbox_triggers;
        impl->joystick = nullptr;
        impl->gamepad = nullptr;
    }
    if (had_engine) {
        SendNeutralPacket(joystick);
    }
    if (impl->restore_trigger) {
        impl->restore_trigger = false;
        for (const bool is_right : {false, true}) {
            const auto block = HdTriggerBlock(is_right, nullptr);
            SDL_SendJoystickEffect(joystick, block.data(), static_cast<int>(block.size()));
        }
    }
    if (impl->restore_speaker) {
        impl->restore_speaker = false;
        const auto block = HdSpeakerBlock(false);
        SDL_SendJoystickEffect(joystick, block.data(), static_cast<int>(block.size()));
    }
    if (had_xbox) {
        SDL_RumbleJoystick(joystick, 0, 0, 0);
        if (triggers) {
            SDL_RumbleJoystickTriggers(joystick, 0, 0, 0);
        }
    }
}

bool RenHDrumPad::IsActive() const {
    std::scoped_lock lock{impl->mutex};
    return impl->engine || impl->xbox ||
           (impl->dsp && (impl->audio_stream || impl->bt_running.load()));
}

bool RenHDrumPad::Vibrate(std::int64_t now_us, bool is_right,
                          const Common::Input::VibrationStatus& status) {
    Impl::SettingsChange change{};
    SDL_Joystick* joystick = nullptr;
    bool handled = false;
    bool engine_sample = false;
    bool send_direct = false;
    std::array<std::uint16_t, 2> direct{};
    bool send_trigger = false;
    std::array<std::uint16_t, 2> trigger{};
    {
        std::scoped_lock lock{impl->mutex};
        if (!impl->joystick) {
            return false;
        }
        joystick = impl->joystick;
        change = impl->RefreshUserSettings();
        const auto& u = impl->user;
        const auto& api = Api();
        // Direct-mode kick state only means something while Direct owns this pad's output;
        // leaving it behind lets Tick settle a stale level over whichever path took over
        if (!(impl->xbox && u.hd_enabled && u.xbox_enabled && u.xbox_direct)) {
            impl->kick_until_us = 0;
        }
        // Guest Switch 2 Mode (commands 213 / 214): the engine picks its translation law from
        // it. One process-wide selector, so remember the last value ANY pad pushed: with a
        // per-pad cache, a second pad's unchanged value would skip the push and leave the
        // selector on the first pad's mode.
        if (api.set_nx2_mode) {
            const int mode = Settings::values.nx2_active_mode.load(std::memory_order_relaxed);
            static std::mutex nx2_mode_mutex;
            static int last_pushed_mode = -1;
            std::scoped_lock nx2_lock{nx2_mode_mutex};
            if (mode != last_pushed_mode) {
                api.set_nx2_mode(mode);
                last_pushed_mode = mode;
            }
        }
        if (impl->xbox && u.hd_enabled && u.xbox_enabled && u.xbox_direct) {
            // Both halves share one pad: each motor follows the stronger side
            const std::size_t side = is_right ? 1 : 0;
            impl->direct_low[side] = ShapeAmplitude(status.low_amplitude, u);
            impl->direct_high[side] = ShapeAmplitude(status.high_amplitude, u);
            const float scale = static_cast<float>(u.xbox_strength) / 100.0f;
            const float mix = static_cast<float>(u.xbox_detail_mix) / 100.0f;
            const auto to_motor = [scale](float v) {
                return static_cast<std::uint16_t>((std::min)(v * scale, 1.0f) * 65535.0f);
            };
            const float low = (std::max)(impl->direct_low[0], impl->direct_low[1]);
            const float high = (std::max)(impl->direct_high[0], impl->direct_high[1]);
            // The big motor also carries part of the fine vibration, where it is felt
            const std::array<std::uint16_t, 2> target{to_motor((std::max)(low, high * mix)),
                                                      to_motor(high)};
            // A sudden rise gets a short burst so the motor spins up in time for quick taps
            constexpr int KickRise = 65535 * 15 / 100;
            constexpr std::int64_t KickUs = 25'000;
            // Renew a steady level 500 ms before SDL's rumble effects expire
            constexpr std::int64_t DirectRenewUs = 1'500'000;
            const float accent = static_cast<float>(u.xbox_hit_accents) / 100.0f;
            const auto kicked = [accent](std::uint16_t t) {
                return static_cast<std::uint16_t>(t + (65535.0f - t) * accent);
            };
            // Triggers carry the fine vibration, left side on LT and right side on RT
            const float trigger_mix = static_cast<float>(u.xbox_trigger_mix) / 100.0f;
            const auto on_trigger = [trigger_mix](std::uint16_t v) {
                return static_cast<std::uint16_t>(static_cast<float>(v) * trigger_mix);
            };
            const std::array<std::uint16_t, 2> trigger_full{to_motor(impl->direct_high[0]),
                                                            to_motor(impl->direct_high[1])};
            const std::array<std::uint16_t, 2> trigger_target{on_trigger(trigger_full[0]),
                                                              on_trigger(trigger_full[1])};
            const bool kicking = impl->kick_until_us != 0 && now_us < impl->kick_until_us;
            if (!kicking && accent > 0.0f &&
                (target[0] - impl->direct_target[0] >= KickRise ||
                 target[1] - impl->direct_target[1] >= KickRise)) {
                impl->kick_until_us = now_us + KickUs;
                trigger = {on_trigger(kicked(trigger_full[0])),
                           on_trigger(kicked(trigger_full[1]))};
                send_trigger = true;
                direct = {target[0] - impl->direct_target[0] >= KickRise ? kicked(target[0])
                                                                         : target[0],
                          target[1] - impl->direct_target[1] >= KickRise ? kicked(target[1])
                                                                         : target[1]};
                send_direct = true;
            } else if (kicking) {
                // Settle on the latest target when the kick ends (Tick); a stop ends it now
                if (target[0] == 0 && target[1] == 0) {
                    impl->kick_until_us = 0;
                    direct = target;
                    send_direct = true;
                    trigger = trigger_target;
                    send_trigger = true;
                }
            } else {
                impl->kick_until_us = 0;
                direct = target;
                // Send on change, and renew a steady nonzero level before SDL expires the
                // write it accepted (rumble effects end after 2000 ms)
                send_direct = direct != impl->direct_sent || impl->direct_pending ||
                              (now_us - impl->last_direct_ok_us > DirectRenewUs &&
                               (direct[0] != 0 || direct[1] != 0));
                trigger = trigger_target;
                send_trigger = trigger != impl->trigger_sent || impl->trigger_pending ||
                               (now_us - impl->last_trigger_ok_us > DirectRenewUs &&
                                (trigger[0] != 0 || trigger[1] != 0));
            }
            impl->direct_target = target;
            impl->trigger_target = trigger_target;
            send_trigger = send_trigger && impl->xbox_triggers;
            handled = true;
        } else if (impl->xbox) {
            if (u.hd_enabled && u.xbox_enabled) {
                api.xbox_push_half_sample(impl->xbox, now_us, is_right,
                                          ShapeAmplitude(status.low_amplitude, u),
                                          status.low_frequency,
                                          ShapeAmplitude(status.high_amplitude, u),
                                          status.high_frequency);
                handled = true;
                engine_sample = true;
            }
        } else if (impl->engine) {
            // A pad whose raw packets SDL refuses (kernel-owned evdev) must not claim the
            // sample: claiming it silences the standard rumble too
            if (u.hd_enabled && impl->raw_transport) {
                api.engine_push_half_sample(impl->engine, now_us, is_right,
                                            ShapeAmplitude(status.low_amplitude, u),
                                            status.low_frequency,
                                            ShapeAmplitude(status.high_amplitude, u),
                                            status.high_frequency);
                handled = true;
                engine_sample = true;
            }
        } else if (impl->dsp) {
            if (u.hd_enabled && u.dualsense_enabled) {
                api.dsp_set_half_target(impl->dsp, is_right,
                                         ShapeAmplitude(status.low_amplitude, u),
                                         status.low_frequency,
                                         ShapeAmplitude(status.high_amplitude, u),
                                         status.high_frequency);
                handled = impl->audio_stream != nullptr || impl->bt_running.load();
                const std::size_t side = is_right ? 1 : 0;
                if (impl->tri[side]) {
                    impl->tri_sample[side] = {status.low_amplitude, status.low_frequency,
                                              status.high_amplitude, status.high_frequency};
                    impl->tri_fresh[side] = true;
                }
            } else {
                // Haptics off: silence the voice coils, the standard rumble takes over
                api.dsp_set_half_target(impl->dsp, is_right, 0.0f, status.low_frequency, 0.0f,
                                        status.high_frequency);
            }
        }
    }
    if (send_direct) {
        // A stop always arrives as a zero; the duration only guards a lost one. The sent
        // state is recorded only when SDL accepts the write; a rejection leaves the level
        // pending so the next sample retries.
        bool ok = false;
        {
            std::scoped_lock send_lock{impl->send_mutex};
            ok = SDL_RumbleJoystick(joystick, direct[0], direct[1], 2000);
        }
        std::scoped_lock lock{impl->mutex};
        if (ok) {
            impl->direct_sent = direct;
            impl->direct_pending = false;
            impl->last_direct_ok_us = now_us;
            impl->direct_generation++;
        } else {
            impl->direct_pending = true;
        }
    }
    if (send_trigger) {
        bool ok = false;
        {
            std::scoped_lock send_lock{impl->send_mutex};
            ok = SDL_RumbleJoystickTriggers(joystick, trigger[0], trigger[1], 2000);
        }
        std::scoped_lock lock{impl->mutex};
        if (ok) {
            impl->trigger_sent = trigger;
            impl->trigger_pending = false;
            impl->last_trigger_ok_us = now_us;
        } else {
            impl->trigger_pending = true;
        }
    }
    if (change.stop_engine) {
        SendNeutralPacket(joystick);
    }
    if (change.reconfigure_engine) {
        impl->ConfigureEngine();
    }
    if (change.sync_hd_haptics) {
        impl->SyncHdHaptics();
    }
    if (engine_sample) {
        Tick(now_us);
    }
    return handled;
}

void RenHDrumPad::Tick(std::int64_t now_us) {
    std::array<Uint8, 10> packet{};
    SDL_Joystick* joystick = nullptr;
    bool open_audio = false;
    bool send = false;
    bool send_xbox = false;
    bool triggers = false;
    std::uint32_t strength = 100;
    bool send_settle = false;
    std::uint64_t settle_generation = 0;
    std::array<std::uint16_t, 2> settle{};
    bool send_settle_trigger = false;
    std::array<std::uint16_t, 2> settle_trigger{};
    std::array<std::uint16_t, 4> motors{};
    {
        std::scoped_lock lock{impl->mutex};
        if (!impl->joystick) {
            return;
        }
        joystick = impl->joystick;
        if (impl->dsp && !impl->audio_stream && !impl->audio_opening &&
            impl->audio_attempts < DualSenseAudioMaxAttempts && now_us >= impl->audio_next_us) {
            impl->audio_next_us = now_us + 2'000'000;
            open_audio = true;
        }
        send = impl->engine && Api().engine_tick(impl->engine, now_us, packet.data()) == 1;
        if (impl->kick_until_us != 0 && now_us >= impl->kick_until_us) {
            impl->kick_until_us = 0;
            settle = impl->direct_target;
            send_settle = settle != impl->direct_sent;
            impl->direct_sent = settle;
            settle_trigger = impl->trigger_target;
            send_settle_trigger =
                impl->xbox_triggers && settle_trigger != impl->trigger_sent;
            if (send_settle_trigger) {
                impl->trigger_sent = settle_trigger;
            }
            settle_generation = impl->direct_generation;
        }
        send_xbox = impl->xbox && !impl->user.xbox_direct &&
                    Api().xbox_tick(impl->xbox, now_us, motors.data()) == 1;
        triggers = impl->xbox_triggers;
        strength = static_cast<std::uint32_t>(impl->user.xbox_strength);
    }
    if (send_settle) {
        // A Direct write that happened while this settle was captured supersedes it: drop
        // the stale level instead of laying it over the newer one
        std::scoped_lock send_lock{impl->send_mutex};
        bool superseded = false;
        {
            std::scoped_lock lock{impl->mutex};
            superseded = impl->direct_generation != settle_generation;
        }
        if (!superseded) {
            const bool ok = SDL_RumbleJoystick(joystick, settle[0], settle[1], 2000);
            std::scoped_lock lock{impl->mutex};
            if (ok) {
                impl->direct_pending = false;
                impl->last_direct_ok_us = now_us;
            } else {
                impl->direct_pending = true;
            }
        }
    }
    if (send_settle_trigger) {
        std::scoped_lock send_lock{impl->send_mutex};
        bool superseded = false;
        {
            std::scoped_lock lock{impl->mutex};
            superseded = impl->direct_generation != settle_generation;
        }
        if (!superseded) {
            const bool ok =
                SDL_RumbleJoystickTriggers(joystick, settle_trigger[0], settle_trigger[1], 2000);
            std::scoped_lock lock{impl->mutex};
            if (ok) {
                impl->trigger_pending = false;
                impl->last_trigger_ok_us = now_us;
            } else {
                impl->trigger_pending = true;
            }
        }
    }
    if (send_xbox) {
        // RenHDrum's mix tops out near 10 % of the range (tuned on the Xbox One S motors);
        // weaker motors need more.
        for (auto& motor : motors) {
            const std::uint32_t scaled = static_cast<std::uint32_t>(motor) * strength / 100u;
            motor = static_cast<std::uint16_t>(std::min<std::uint32_t>(65535u, scaled));
        }
        // Grips last longer than a tick; trigger pulses stay short so a lost stop cannot
        // leave those motors running.
        const bool sent = SDL_RumbleJoystick(joystick, motors[0], motors[1], 2000);
        if (triggers) {
            SDL_RumbleJoystickTriggers(joystick, motors[2], motors[3], 200);
        }
        if (!sent) {
            // Some paths refuse rumble until the first button press: retry on the next tick
            std::scoped_lock lock{impl->mutex};
            if (impl->xbox) {
                Api().xbox_resend(impl->xbox);
            }
        }
    }
    if (open_audio) {
        impl->OpenDualSenseAudio();
    }
    // HD Haptics: one trigger-kick step per side; with no fresh sample the core falls back to off
    impl->TriggerStep(now_us, false);
    impl->TriggerStep(now_us, true);
    if (send) {
        if (!SDL_SendJoystickEffect(joystick, packet.data(), static_cast<int>(packet.size())) &&
            now_us >= impl->next_effect_error_us.load(std::memory_order_relaxed)) {
            // Stock SDL's evdev backend answers every raw packet with SDL_Unsupported; a pad that
            // the kernel's hid-nintendo driver owns lands there, and Bluetooth is the usual way
            // that happens. Silence here reads as "the engine does nothing", so say it once in
            // a while instead.
            impl->next_effect_error_us.store(now_us + 10'000'000, std::memory_order_relaxed);
            LOG_WARNING(Input, "RenHDrum: the controller refused the rumble packet: {}",
                        SDL_GetError());
        }
    }
}

} // namespace InputCommon

#else // ENABLE_RENHDRUM

namespace InputCommon {

bool IsRenHDrumBuilt() {
    return false;
}

bool IsRenHDrumLoaded() {
    return false;
}

int GetRenHDrumConnectedKind() {
    return -1;
}

int GetRenHDrumXboxTriggers() {
    return -1;
}

bool GetRenHDrumXboxNeedsRawInput() {
    return false;
}

std::vector<RenHDrumOption> GetRenHDrumOptions() {
    return FallbackRenHDrumOptions();
}

struct RenHDrumPad::Impl {};

std::shared_ptr<RenHDrumPad> RenHDrumPad::Create(SDL_Joystick*, SDL_Gamepad*) {
    return nullptr;
}

RenHDrumPad::RenHDrumPad(std::unique_ptr<Impl> impl_) : impl{std::move(impl_)} {}
RenHDrumPad::~RenHDrumPad() = default;
void RenHDrumPad::Shutdown() {}
bool RenHDrumPad::IsActive() const {
    return false;
}
bool RenHDrumPad::Vibrate(std::int64_t, bool, const Common::Input::VibrationStatus&) {
    return false;
}
void RenHDrumPad::Tick(std::int64_t) {}

} // namespace InputCommon

#endif // ENABLE_RENHDRUM
