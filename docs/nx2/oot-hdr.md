# Ocarina of Time: HDR output from the game's scene (prototype)

`EDEN_NX2_OOT_HDR=1` (default off) presents The Legend of Zelda: Ocarina of Time
(`0400486024DEC000`) in HDR10 from the game's own linear scene, instead of its SDR tonemap or its
built-in HDR mode. It needs the Vulkan renderer, the temporal 4K path with FSR 4
(`ocarina_fsr`, Max Resolution add-on, `EDEN_NX2_FSR4_LIB`) and a surface that offers
`VK_COLOR_SPACE_HDR10_ST2084_EXT` with a 10-bit format (`A2B10G10R10_UNORM_PACK32` or
`A2R10G10B10_UNORM_PACK32`; under gamescope: HDR on in Steam, or `--hdr-enabled`).

## HDR or SDR: chosen once, at the first use

The switch can be left on. The mode is chosen once per process, the first time the game needs
it (building its tonemap pipeline, or creating the upscaler), from the latest swapchain: **HDR**
when that swapchain is HDR10, otherwise **SDR**, which is the normal path (the game's tonemap,
FSR's SDR mode, the configured window filter, never an HDR10 swapchain). The log has one line:
`Ocarina HDR: HDR mode (swapchain ...)` or `Ocarina HDR: SDR mode, the surface offers no HDR10
(swapchain ...)`.

Not earlier, because the game's pipelines load from the disk cache before the window has its real
surface: gamescope first gives a small startup window (100x30) that offers SDR formats only, even
with HDR on, and the HDR10 swapchain comes seconds later. While the mode is undecided, the boot
pipeline build leaves out the pipelines with a tonemap shader (`Ocarina HDR: pipelines with
tonemap ... are built at their first draw`); their first draw builds them in the chosen mode.
Until then a swapchain takes HDR10 whenever the surface offers it (the HDR10 present pass shows
SDR sources at UI white). If HDR is turned off in Steam after the mode chose HDR, the swapchain
becomes SDR and highlights clip at white (logged once); the mode doesn't change until a restart.

## What the game does

Its tonemap pass (fragment shader `bded5ce917183214`; `3ac011bd2d772994` and
`6cb06edf14be1d79` are listed as other variants) reads the linear scene and a bloom texture,
applies a colour matrix from its constant buffer (c3, words 0 to 11), adds bloom raised to a
power (word 13), scales by an exposure value (word 14) and maps each channel with the widely
published ACES filmic fit (Narkowicz), clamped to 0..1. There is no LUT. The output (B10G11R11,
linear, 0..1) feeds a compute pass that stores the square root of the colour for the game's
upscaler (replaced by FSR here), then effects and HUD, then an 8-bit sRGB present image.

## What HDR mode changes

- **Tonemap** (`vk_ocarina_hdr.cpp`, `ocarina_hdr_tonemap.frag`): the shader is replaced at
  SPIR-V emission (monolithic and library pipelines) by one with the same maths up to a knee
  (the game's output 0.5 by default), then a C1 shoulder `y_k + (P - y_k) * t / sqrt(1 + t^2)`
  up to the peak. The replacement takes over the game shader's descriptor bindings and is used
  only when the translated shader has exactly that interface (c3 words < 16, two textures, the
  ACES constants); otherwise the game's shader stays and the log says so.
- **FSR 4** runs with `FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE` on linear colour (the guest's
  sqrt-encoded input squared); its output goes back to the guest linear, unclamped.
- **4K composite** (`ocarina_hdr_composite.comp`, RGBA16F): HUD extraction as in SDR; the scene
  part keeps its highlights. Values are SDR-style encoded relative to UI white (above 1 for
  highlights), so screenshots and an SDR swapchain still show a normal SDR picture.
- **Presentation**: HDR10 swapchain; the window adaptation (`vulkan_present_hdr10.frag`) decodes
  every source (gamma 2.2 or the sRGB curve), scales to the UI white, converts BT.709 to
  BT.2020 and encodes PQ. Frames without the 4K composite (menus, fallbacks) are SDR at UI white.

## Settings (environment, read once)

| Variable | Default | Meaning |
|---|---|---|
| `EDEN_NX2_OOT_HDR` | 0 | 1: HDR when the surface offers HDR10, otherwise the normal SDR path (see above) |
| `EDEN_NX2_OOT_HDR_PEAK` | 1500 | display peak, nits |
| `EDEN_NX2_OOT_HDR_PAPER` | 203 | game brightness: nits of the game's SDR white |
| `EDEN_NX2_OOT_HDR_UI` | 203 | HUD/menu white, nits |
| `EDEN_NX2_OOT_HDR_GAMMA` | 2.2 | decode SDR as a 2.2 display does (0: sRGB curve) |
| `EDEN_NX2_OOT_HDR_KNEE` | 0.5 | game output level where the HDR curve leaves the game's |
| `EDEN_NX2_OOT_HDR_STATS` | 0 | 1: log output luminance percentiles every 600 frames |
| `EDEN_NX2_OOT_HDR_DUMP` | | file: write one 4K composite (RGBA16F, encoded) |

## Limits

- Needs FSR 4: with FSR 3.1 (or a declined frame) the game's own upscale runs on the HDR tonemap
  output and its 8-bit present clips highlights hard.
- No `VK_EXT_hdr_metadata` yet (gamescope then uses its defaults for the HDMI infoframe).
- Shader hashes and constant layout are those of the game build in `oot_network.h`.
