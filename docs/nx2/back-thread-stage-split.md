# Splitting the GPU back thread: feasibility

Measured on Donkey Kong Bananza (040074A01BF12000), "Cave-In Site" with the characters in view,
i7-10700F and GTX 1650 SUPER, nx2 `bfa046dcf2` plus the padded target, write extent, list copy
and blit changes, built with the refreshed `lichii-windows.profdata`. Nothing here is implemented.

## Where the frame goes

About 12,300 draws a frame. The GPU back thread is idle 1.4 ms of a 33 ms frame
(`EDEN_NX2_BACK_IDLE_TRACE`), the front waits for it 54% of the time, the Vulkan worker is idle
80% and the host GPU 55%. The frame time is the back thread's CPU time.

Sampled share of the back thread:

| Work | Share |
|---|---|
| Buffer cache per draw: `UpdateGraphicsBuffers`, `BindHostGeometryBuffers`, `BindHostStageBuffers`, storage binds | 35% |
| Rest of the draw: pipeline, image descriptors, render targets, dynamic state, recording | 33% |
| Register writes, constant buffer words, ring decode | 14% |
| Compute, waits, everything else | 18% |

No function is above about 10% self time, and the per-draw path is already a delta path for 75%
of the draws.

## Why a stage cannot simply run late

The cut that pays is the buffer cache half of a draw. A stage that binds buffers some draws
behind the thread that applies state reads guest memory later than the stream position it works
for:

- uniform data is copied from guest memory when it is bound
  (`BufferCache::BindHostGraphicsUniformBuffer`);
- CPU-modified ranges of vertex, index, uniform and storage bindings are uploaded from guest
  memory when they are bound (`SynchronizeBuffer`);
- the stream itself writes that memory: inline constant buffer data
  (`Maxwell3D::ProcessCBMultiData` -> `AccelerateInlineToMemory`), 0.46 writes a draw here, which
  the earlier thread also needs at once, to read texture handles from the same constant buffers.

Two threads at different stream positions would need two versions of that memory. The split's
front has the same problem and solves it by noting writes and draining on a conflict.

## Is the conflict rare enough?

A temporary probe kept each draw's bound ranges (17.7 a draw) and checked every inline write
against the ranges of the 16 draws before it:

| View | Draws in 10 s | Inline writes | Writes inside a range bound by the draw before | Kind |
|---|---|---|---|---|
| Characters visible | 3,602,822 | 1,670,735 | 101,170 (2.8% of draws) | storage only |
| Characters hidden | 1,415,363 | 1,037,872 | 3,600 (0.25% of draws) | storage only |

No write touched a uniform, vertex or index range of the 16 draws before it: the game allocates
them from a ring. Every conflict is with a storage binding, which a native pointer makes up to
8 MiB long. So a binding stage that drains on a conflict would run undisturbed for 97% of the
draws in this title. One title is not a rule: a game that rewrites a constant buffer in place
conflicts on every draw and gets no gain, but stays correct.

## A design that needs no memory versions

At most one draw is outstanding, and the thread that owns the caches settles it before it does
anything else with them:

1. Draw k, back thread: pipeline, textures, render targets and dynamic state as today. The buffer
   half becomes a job: the uniform and storage bind calls since the last draw, the vertex stream
   and index buffer registers the cache reads (it reads only `vertex_streams`,
   `vertex_stream_limits`, the native stream sizes, `draw_state`, three dirty flag groups, the
   constant buffer addresses and the transform feedback registers), and the storage pointer words.
2. A helper thread runs the job under the buffer cache's lock: it resolves, synchronizes and
   touches the buffers, makes the uniform keep decisions and stream copies, and stores what it
   would have bound in the job instead of recording it.
3. The back goes on with the register writes up to the next draw. Before the next draw, any other
   rasterizer entry, any buffer cache lock it takes and any inline write that touches a range of
   the job, it waits for the job, then records the binds, the descriptor update and the draw.

Nothing reads memory at a later stream position than today, so nothing needs a second version of
it. What the helper cannot do without the scheduler or the staging pool (creating or joining a
buffer, an upload) it must find before it changes anything, and leave the draw to the back.

What it touches: the buffer cache's bind functions (a snapshot instead of live registers, a sink
instead of the runtime), `BufferCacheRuntime`'s bind calls, the descriptor queue (slots reserved
for the buffer entries), both `Configure` paths, the delta draw path and uniform keeps, and every
rasterizer entry point (108 buffer cache call sites, 65 methods, 34 lock sites in
`vk_rasterizer.cpp` alone, which is why the settle belongs in the lock).

## What it would give

The helper can take the cache logic, not the recording, which stays in stream order on the back:
about 20% of the back thread, less 5% to 6% for capturing the job and replaying its results.
The expected gain is 12% to 15% of the frame in scenes bound by the back thread (an upper bound
of 24% if recording moved too), and nothing where the back thread is not the limit.

A fork and join inside one draw (no outstanding draw at all) was estimated from the profile zones
at about 8%: in a delta draw there is little for the back to do while the helper binds.

## Why it is not in this change

Each step before the last gives no speed-up, and the last one changes how every Switch 2 title
binds its buffers and updates its descriptors. On the PC used here only Bananza reaches gameplay
(Ocarina of Time loses the Vulkan device while it loads a save, with or without changes), so the
change could not be checked against a second title. It needs a verify mode like
`EDEN_NX2_FAST_DRAW_VERIFY` (run the job on the back too and compare) before it is turned on.
