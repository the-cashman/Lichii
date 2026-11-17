# Fire Emblem: Fortune's Weave 60 FPS mode

`Fire Emblem: Fortune's Weave: 60 FPS mode` writes 60 into the game's own target frame rate.
Gameplay then runs at 60 FPS with its normal speed, instead of the game's shipped 30 FPS.

## The game's target frame rate

Fortune's Weave's graphics object is found through a global pointer and holds:

| Offset | Meaning | Shipped value |
|---|---|---|
| `+0x1d8` | display rate | 60 |
| `+0x1dc` | target frame rate | 60 in the title and menus, 30 in gameplay |
| `+0x1e0` | longest present interval | 5 |

The present asks for `min(+0x1e0, display / min(display, target))`, and 59 readers step the
game's logic by `60.0 / target` per frame. At 30 that is an interval of 2 vsyncs and a step of
2.0; at 60 it is an interval of 1 and a step of 1.0. Writing 60 therefore gives 60 FPS without
the 2× speedup of unlocking the frame rate (`Ctrl+U` / `EDEN_NX2_SWAP_INTERVAL`).

This is the mechanism nxnative documents for the title; its `dm/tools/fps_target.py` finds the
field by its readers rather than by a build address. The field was verified on build
`DB93B0125F485060` (v131072), where the global is `main+0x21e60d8`, the field is `+0x1dc`, and
gameplay sets 30 at `main+0x7bf1c0`.

## How other games' dynamic modes work

The emulator already has three shapes of game-specific frame-rate control:

- **Ocarina of Time** uses a built-in Add-On that replaces the game's frame-delta function with
  a routine returning real time, so the game itself can run at any frame rate. See
  [`ocarina-dynamic-fps.md`](ocarina-dynamic-fps.md).
- **Mario Kart World** has a native 1- or 2-vblank mode word. The loader patches the mode load
  to a fixed value, or to a stub whose word the present path switches between 1 and 2. See
  `src/core/loader/nso.cpp` and `UpdateMkwFrameMode` in `src/core/hle/service/nx2_gsv.h`.
- **Hyrule Warriors: Age of Imprisonment** reads a frame-interval global (1 or 2). The loader
  finds it, and the present path writes 1 or 2. See
  [`nx2_frame_interval.h`](../../src/core/hle/service/nx2_frame_interval.h).

Fortune's Weave is the third shape: the game owns the decision, and the emulator only writes its
target frame rate, as nxnative's host does. No game code is patched.

## Implementation

1. **Loader** (`src/core/loader/fps_target.h`, called from `nso.cpp`): scans `main.text` for the
   field's readers. A reader is

   ```
   ldr  sT, [xB, #field]
   scvtf sC, sT
   fdiv sD, sM, sC      ; sM = 60.0f, made by movz wK, #0x4270, lsl #16; fmov sM, wK
   ```

   where `xB` is loaded from a global (`ldr xB, [xA]`, `xA` made by `adrp` + `add`, or `adrp` +
   `ldr` through its GOT slot). A field with at least eight such readers and a store of an
   immediate 30 is the target; exactly one field must qualify. The scan needs no build ID and
   should survive a game update that keeps the engine. For build `DB93B0125F485060` the known
   offsets are preferred over the scan's result.

2. **Present path** (`UpdateFortuneWeaveFrameTarget` in `nx2_gsv.h`): after each present, reads
   the object pointer and the current target, remembers the value the game wrote when it changes
   mode, and writes the emulator's target when it differs:
   - 60 with the fallback off;
   - 60, or 30 when frames stop fitting, with the fallback on.

   The 30/60 choice comes from the existing `Nx2FrameInterval::Controller`, which measures the
   time between presents and already drives Hyrule Warriors' dynamic FPS. Fortune's Weave
   constructs it with a 1.12 drop threshold (about 53.5 FPS) because its logic steps by the
   target: a frame rate 10–17% short is a 10–17% slowdown. At 30 the game presents every second
   vblank and steps 2.0, so a slow PC keeps normal game speed instead of slow motion. Turning
   the mode off writes the game's own value back.

   **Why 1.12 rather than Hyrule's 1.2:** 1.2 is 50 FPS. At 50–53 FPS this game's logic runs
   10–17% slow, because its step is `60 / target`, while 30 FPS with step 2.0 is normal speed.
   Fortune's Weave is turn-based, so it prefers the correct speed. 1.12 is also the mean a probe
   must stay under (`KeepAtOrBelow`); a drop threshold below it would drop, probe back into the
   band, and drop again. Hyrule Warriors keeps 1.2 because its action gameplay values the extra
   20 FPS over the 10–17% slowdown.

3. **Settings** (Graphics > Advanced):
   - `Fire Emblem: Fortune's Weave: 60 FPS mode` (`fe_60fps`, off by default);
   - `...: drop to 30 FPS when slow` (`fe_60fps_fallback`, on by default, only with the 60 FPS
     mode).

## Limits

- The scan follows nxnative's pattern; on v131072 it found 57 readers and the store of 30
  (nxnative reports 55 of 59 with its narrower window), which is enough. A build that changes
  the pattern and is not `DB93B0125F485060` logs `target frame rate not found` and leaves the
  game at its own rate.
- The game steps by its target whatever the real frame time, so with the fallback off a PC that
  can't hold 60 runs in slow motion. The fallback is the safe default. It measures about a
  second of presents before it drops, so a scene that becomes heavy at once can play briefly in
  slow motion.
- 120 FPS is not implemented. The game would also need its display rate and present interval
  changed; nxnative does this through its general frame-step controller, not this field alone.
- Movies are not capped separately. nxnative's tests with this field at 60 played the AV1
  movies; if a cutscene's picture rate is lower than the target, the scene may play at the wrong
  speed. Report it with a log if seen.

## Measured

On an RTX 4070 Ti at 1× in the first battle map: 60.0009 FPS over 300 presents (16.6 ms, range
14–19 ms), with the log's 600-present windows at 0 of 600 at 30 FPS. The 1× title screen held 60
after its shaders compiled; its heavier loading moments dropped to 30 and returned to 60.
