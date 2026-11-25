// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <SDL3/SDL.h>

#include "common/input.h"

namespace InputCommon {

/// True when the build includes RenHDrum support (ENABLE_RENHDRUM).
bool IsRenHDrumBuilt();

/// True when renhdrum.dll (librenhdrum.so / .dylib) was found and loaded. Loads it on first call.
bool IsRenHDrumLoaded();

/// Kind of the controller RenHDrum connected most recently and still drives: 0 Switch
/// controller, 1 DualSense, 2 Xbox style, or -1 when none.
int GetRenHDrumConnectedKind();

/// For connected Xbox-style controllers: 1 when one exposes trigger motors, 0 when none does,
/// -1 when no Xbox-style controller is connected.
int GetRenHDrumXboxTriggers();

/// True when an official Xbox controller is connected but its trigger motors are out of reach:
/// on Windows SDL drives them only through raw input ("Enable XInput 8 player support").
bool GetRenHDrumXboxNeedsRawInput();

/// One row of RenHDrum's settings table, as the loaded library publishes it. The dialog builds
/// its rows from this so every option appears exactly as the engine knows it.
struct RenHDrumOption {
    std::string key;
    std::string label;
    std::string help;
    std::vector<std::string> choices;
    int kind{};         // 0 bool, 1 int, 2 choice (value = index)
    int min{};
    int max{};
    int step{};
    int def{};
    unsigned devices{}; // bit 1 Switch, 2 DualSense, 4 Xbox, 8 Steam
    int tier{};         // 0 basic, 1 advanced
    int link{};         // 0 any, 1 wired only, 2 Bluetooth only
    int requires_row{}; // row index that must be non-zero for this row to matter, -1 none
};

/// The library's settings table. When the library predates it (pre-0.5.0), a copy of the 0.5.0
/// table stands in so the dialog keeps working; the values still bind to Eglantine's settings.
std::vector<RenHDrumOption> GetRenHDrumOptions();

/**
 * Vibration for one SDL controller through RenHDrum (https://github.com/trzz1214/RenHDrum):
 * Switch Pro Controller / Joy-Con get HD Rumble packets, a wired DualSense gets audio haptics on
 * its voice coils. Settings come from Settings::values and are re-read on every vibration.
 */
class RenHDrumPad {
public:
    /// nullptr when RenHDrum is not loaded or the controller is not one it drives.
    static std::shared_ptr<RenHDrumPad> Create(SDL_Joystick* joystick, SDL_Gamepad* gamepad);

    ~RenHDrumPad();

    RenHDrumPad(const RenHDrumPad&) = delete;
    RenHDrumPad& operator=(const RenHDrumPad&) = delete;

    /// Releases the engine and the audio stream. Call before the SDL joystick is closed.
    void Shutdown();

    /// True while RenHDrum plays this controller's vibration.
    bool IsActive() const;

    /// HLE thread. True when RenHDrum took the vibration; false means use the standard rumble.
    bool Vibrate(std::int64_t now_us, bool is_right, const Common::Input::VibrationStatus& status);

    /// Input thread, about once per millisecond: due HD Rumble packets and DualSense audio retries.
    void Tick(std::int64_t now_us);

private:
    struct Impl;
    explicit RenHDrumPad(std::unique_ptr<Impl> impl_);

    std::unique_ptr<Impl> impl;
};

} // namespace InputCommon
