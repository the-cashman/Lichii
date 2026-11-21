// From Banana Native (github.com/hugorosario/BananaNative, Android port of github.com/seedds/nxnative by seedds), MIT: see LICENSE in src/video_core/nvn_direct/bananative.
// NVN interface layer: takes over the nvnBootstrapLoader imported by main; the NVN driver in sdk is not executed at all.
// All 523 NVN names return counting thunks (nvn_thunks.S), and the thunks forward to the implementations here:
//   - those with semantics (device / memory pools / textures and samplers / queue / window / sync / events / command buffer memory) are implemented per NVN semantics, and the objects are real;
//   - those that only change GPU state and have no CPU-visible result (state objects, bindings, draws, clears...) are recorded as commands and replayed at submit;
//   - the rest (the 307 that main only loads but never calls + those without a prototype in the header) go to nvn_unimpl_entry: reports a failure and prints the name when called.
// GPU address = CPU address (memory pools use the real memory given by the game, also built as zero-copy MTLBuffer). Commands are recorded into the command memory given by the game and replayed at submit into a
// Metal command buffer (nvn_mtl.m): buffer copies / uniform updates go through blit (GPU ordering), event value writes / counters are written after the GPU completes;
// sync object = submit fence value, and waits are released only after the GPU completes.
// Every implementation gets a compile-time signature check against the prototypes in nvn_protos.h (NVNboolean parameters are received as uint32_t, only the low 8 bits are looked at).
// Environment switches: DM_NVN_FRAMES=N prints the call table per frame for the first N frames (default 3); DM_NVN_TSV=path writes the per-frame call table and timing to a TSV (DM_NVN_TSV_CALLS=0 writes timing only); DM_SLOWFRAME=1 writes only the per-frame breakdown of dropped frames to the log; DM_NVN_LOG=1 logs object creation.
#include "../../host/hle.h"
#include "nvn_protos.h"
#include "nvn_count.h"
#include "nvn_tex.h"
#include "../tex/nvn_deswizzle.h"
#include "nvn_mtl.h"
#include "nvn_pace.h"
#include "nvn_fstep.h"
#include "nvn_timestamp.h"
#include "../../host/front.h"
#include "nvn_dirty.h"
#include "../../../../nvn_direct_front.h"   // Lichii: the guest call site (nvn_caller)
#if PLAT_APPLE
#include <mach/mach.h>
#include <mach/thread_policy.h>
#include <sys/sysctl.h>
#endif
#include <unistd.h>
#include <sys/mman.h>
#include <errno.h>
#if PLAT_MAC
#define DM_MEM_CAP_DEFAULT_GB 24
#else
#define DM_MEM_CAP_DEFAULT_GB 0
#endif
#include <sys/resource.h>
#include <assert.h>
#include <stdatomic.h>

// ---------- Slot table, thunks, unimplemented stubs ----------
typedef struct { _Atomic uint64_t cnt; void *fn; } nvn_slot_t;
nvn_slot_t nvn_slot[NVN_NFUNC];                 // The thunk accesses it by i*16; the layout must not change
_Static_assert(sizeof(nvn_slot_t) == 16, "the trampolines are addressed in 16-byte steps");
extern void *nvn_thunk[NVN_NFUNC];
extern const char *nvn_names[NVN_NFUNC];
extern void nvn_unimpl_entry(void);
extern void nvn_prof_entry(void);
extern void nvn_big_entry(void);
void *nvn_big_real[NVN_NFUNC];            // the real implementation of functions that run on the big host stack (nvn_entry.S nvn_big_entry)
void *nvn_real[NVN_NFUNC];               // DM_NVN_PROF: the real implementation called by the timing wrapper
_Atomic uint64_t nvn_prof_ticks[NVN_NFUNC]; // DM_NVN_PROF: cumulative time of each function (cntvct counts)
static uint8_t g_unimpl_seen[NVN_NFUNC];

uint64_t nvn_unimpl_c(nvn_slot_t *slot, uint64_t x0, uint64_t x1, uint64_t x2) {
    unsigned i = (unsigned)(slot - nvn_slot);
    if (!g_unimpl_seen[i]++) LOG("[RED] NVN unimplemented function called: %s (x0=%#llx x1=%#llx x2=%#llx)", nvn_names[i], x0, x1, x2);
    return 0;
}

typedef struct { const char *name; void *fn; } nvn_impl;
#define NVN(name, ret, ...)                                                                       \
    static ret h_##name(__VA_ARGS__);                                                             \
    static const NVNT_##name chk_##name __attribute__((unused)) = h_##name;                       \
    PLAT_TABLE(swnvn) static const nvn_impl e_##name = {#name, (void *)h_##name}; \
    static ret h_##name(__VA_ARGS__)
// Newer interfaces without prototypes: no signature check
#define NVN_NOPROTO(name, ret, ...)                                                               \
    static ret h_##name(__VA_ARGS__);                                                             \
    PLAT_TABLE(swnvn) static const nvn_impl e_##name = {#name, (void *)h_##name}; \
    static ret h_##name(__VA_ARGS__)

static int g_log;
static pthread_mutex_t g_sub_mu = PTHREAD_MUTEX_INITIALIZER;
static _Atomic uint64_t g_serial;
uint64_t nvn_new_serial(void) { return ++g_serial; }
void nvn_serial_floor(uint64_t s) { uint64_t c = atomic_load(&g_serial); while (c < s && !atomic_compare_exchange_weak(&g_serial, &c, s)) {} }
static unsigned g_nsem;                          // Number of semantic implementations (filled by nvn_init_slots)
#define OLOG(...) do { if (g_log) LOG("[nvn] " __VA_ARGS__); } while (0)

static void nvn_init_slots(void) {
    static int done; if (done) return; done = 1;
    g_log = getenv("DM_NVN_LOG") != NULL;
    unsigned long sz;
    const nvn_impl *im = (const nvn_impl *)plat_table("swnvn", &sz);
    unsigned nim = im ? (unsigned)(sz / sizeof *im) : 0, hit = 0;
    for (unsigned i = 0; i < NVN_NFUNC; i++) nvn_slot[i].fn = (void *)nvn_unimpl_entry;
    for (unsigned k = 0; k < nim; k++) {
        if (!im[k].name) continue;   // Lichii: padding a COFF linker left between the table's pieces (plat_windows.c)
        unsigned i = 0;
        while (i < NVN_NFUNC && strcmp(nvn_names[i], im[k].name)) i++;
        if (i == NVN_NFUNC) die("NVN implementation %s is not in main's name table", im[k].name);
        nvn_slot[i].fn = im[k].fn; hit++;
    }
    g_nsem = hit;
}

static void *nvn_lookup(const char *name) {
    nvn_init_slots();
    for (unsigned i = 0; i < NVN_NFUNC; i++)
        if (!strcmp(nvn_names[i], name)) return nvn_thunk[i];
    LOG("[RED] NVN address lookup of an unknown name: %s", name ? name : "(null)");
    return NULL;
}

// nvnBootstrapLoader is imported by main (defined in sdk); after it is forcibly taken over, the NVN driver in sdk is no longer referenced
HLE_FORCE("nvnBootstrapLoader", "the NVN graphics API is replaced wholesale with a host null device (Metal behind it); the SDK's NVN driver does not run",
          void *, h_nvnBootstrapLoader, const char *name) { return nvn_lookup(name); }

// ---------- Boolean parameters: Switch callers do not extend them, only look at the low 8 bits ----------
#define B(x) ((x) & 0xff)

#include "nvn_obj.h"
#include "nvn_capture.h"   // DM_NVN_CAPTURE hooks

// ---------- Device ----------
NVN(nvnDeviceBuilderSetDefaults, void, NVNdeviceBuilder *b) { memset(b, 0, sizeof(dev_builder_t)); }
NVN(nvnDeviceBuilderSetFlags, void, NVNdeviceBuilder *b, int f) { ((dev_builder_t *)b)->flags = f; }
NVN(nvnDeviceInitialize, NVNboolean, NVNdevice *d, const NVNdeviceBuilder *b) {
    dev_t_ *o = (dev_t_ *)d; memset(o, 0, sizeof *o);
    o->magic = MAGIC('D', 'E', 'V', 0); o->flags = ((const dev_builder_t *)b)->flags;
    nvn_gob_switch2 = dm_sdk_ounce();   // the Switch 2's order inside a GOB, for the textures the game tiled itself (nvn_deswizzle.h)
    o->origin = NVN_WINDOW_ORIGIN_MODE_LOWER_LEFT; mtl_set_origin_lower_left(1);   // NVN's default; every game so far switched to upper-left right away, Link's Awakening keeps it
    mtl_set_depth_zero(0);   // NVN's default depth mode: near = -w (GL)
    OLOG("DeviceInitialize %p flags=%#x", (void *)d, o->flags);
    return 1;
}
NVN(nvnDeviceFinalize, void, NVNdevice *d) { ((dev_t_ *)d)->magic = 0; }
NVN(nvnDeviceGetProcAddress, PFNNVNGENERICFUNCPTRPROC, const NVNdevice *d, const char *name) {
    (void)d; return (PFNNVNGENERICFUNCPTRPROC)nvn_lookup(name);
}

static void nvn_api(int api[2]);
static int g_api[2]; static pthread_once_t g_api_once = PTHREAD_ONCE_INIT;
static void api_find(void) { int api[2] = {-1, -1}; nvn_api(api); const char *e = getenv("DM_NVN_API"); if (e) sscanf(e, "%d.%d", &api[0], &api[1]); g_api[0] = api[0]; g_api[1] = api[1]; }
// Device properties: values measured on Tegra X1 (nnheaders nvn_DeviceConstantsNX.h, obtained by calling nvnDeviceGetInteger item by item on SMO)
static const int g_devinfo[] = {
    // API version: 55.15 here, 53.313 for SDK 10 and older (nvn_api below; the profile's device_api overrides): SMBW (SDK 16) requires major == 55 and minor >= 15 (main 0x5a7b48); Super Mario 3D World + Bowser's Fury (SDK 10) requires major == 53 and minor >= 313 (main+0x711c88),
    // and on a mismatch it creates no device and leaves the device pointer null, which crashes in nnMain right after. DM_NVN_API=major.minor overrides
    [0x00] = 55, [0x01] = 15,
    [0x02] = 14, [0x03] = 65536, [0x04] = 256, [0x05] = 8, [0x06] = 16, [0x07] = 4, [0x08] = 16, [0x09] = 32,
    [0x0A] = 16, [0x0B] = 4, [0x0C] = 4, [0x0D] = 4, [0x0E] = 16, [0x0F] = 32, [0x10] = 32, [0x11] = 256, [0x12] = 256,
    [0x13] = 4, [0x14] = 8, [0x15] = 512, [0x16] = 256, [0x17] = 2, [0x18] = 256, [0x19] = 4096, [0x1A] = 131072,
    [0x1B] = 16, [0x1C] = 1536, [0x1D] = 1024, [0x1E] = 64, [0x1F] = 1536, [0x20] = 65535, [0x21] = 65535, [0x22] = 65535,
    [0x23] = 8, [0x24] = 1048576, [0x25] = 4096, [0x26] = 16, [0x27] = 4096, [0x28] = 1, [0x29] = 1, [0x2A] = 1,
    [0x2B] = 262144, [0x2C] = 15, [0x2D] = 2048, [0x2E] = 1, [0x2F] = 1, [0x30] = 14, [0x31] = 5, [0x32] = 1, [0x33] = 8,
    [0x34] = 8, [0x35] = 4, [0x36] = 32, [0x37] = 128, [0x38] = 32, [0x39] = 128, [0x3A] = 65536, [0x3B] = 1, [0x3C] = 4,
    [0x3D] = 16384, [0x3E] = 134217728, [0x3F] = 2048, [0x40] = 16384, [0x41] = 16384, [0x42] = 1, [0x43] = 1, [0x44] = 1,
    [0x45] = 1, [0x46] = 5, [0x47] = 1, [0x48] = 1, [0x49] = 1, [0x4A] = 1, [0x4B] = 1, [0x4C] = 1, [0x4D] = 16,
    [0x4E] = 1024, [0x4F] = 32, [0x50] = 4096, [0x51] = 65536, [0x52] = 65536, [0x53] = 256, [0x54] = 16384,
    [0x55] = 262144, [0x56] = 4096, [0x57] = 1, [0x58] = 4, [0x59] = 2, [0x5A] = 0, [0x5B] = 4096, [0x5C] = 16384,
    [0x5D] = 4096, [0x5E] = 128, [0x5F] = 4096,
};
// The Switch 2's values: NVN2 compiled into the game's own sdk answers from constants (Donkey Kong Bananza, sdk 20.5.19: the switch at sdk+0x5d9960 and its table lookup
// sdk+0x5c0a20, read statically; the switch's own constants, among them the versions, are read from the game's sdk at run time, dm_sdk_nvn2_item, and this table has the newest seen, SDK 21.4.4's).
// The item numbers are NVN's: API 231.39 at 0 / 1 and GPU code 201.0-201.20 at 0x2E-0x31 (0xc9 = the version of its shaders' control sections).
// Ampere differs from the X1 in: 32 vertex bindings and attributes (the host keeps 16, all its tables hold), 1,024 compute threads, an unbounded dispatch width, 16 texture
// levels, 16 KB scratch granularity, 2 KB shader code padding, 32 separate sampler bindings, and items up to 0xA9. DEVINFO_RT: items NVN2 reads from the device's own GPU
// description at run time (scratch scale factors, L2 size, layers, texture sizes, linear stride alignment, pool page size, …), not knowable from the binary: these keep the X1 table's
// value below 0x60, and above it answer 0 with a red line as before
enum { DEVINFO_RT = -1 };
static const int g_devinfo2[] = {
    [0x00] = 231, [0x01] = 39, [0x02] = 14, [0x03] = 65536, [0x04] = 256, [0x05] = 8, [0x06] = 16 /* NVN2 32 */, [0x07] = 4, [0x08] = 16, [0x09] = 32, [0x0A] = 16, [0x0B] = 4,
    [0x0C] = 4, [0x0D] = 4, [0x0E] = 16 /* NVN2 32 */, [0x0F] = 32, [0x10] = 32, [0x11] = 256, [0x12] = 256, [0x13] = 4, [0x14] = 8, [0x15] = 512, [0x16] = 256, [0x17] = DEVINFO_RT,
    [0x18] = DEVINFO_RT, [0x19] = 4096, [0x1A] = 16384, [0x1B] = 16, [0x1C] = 1024, [0x1D] = 1024, [0x1E] = 64, [0x1F] = 1024, [0x20] = 2147483647, [0x21] = 65535, [0x22] = 65535,
    [0x23] = 8, [0x24] = 1048576, [0x25] = 4096, [0x26] = 16, [0x27] = 4096, [0x28] = 1, [0x29] = 1, [0x2A] = 1, [0x2B] = DEVINFO_RT, [0x2C] = 16, [0x2D] = DEVINFO_RT,
    [0x2E] = 201, [0x2F] = 201, [0x30] = 20, [0x31] = 0, [0x32] = 1, [0x33] = 8, [0x34] = 8, [0x35] = 4, [0x36] = 32, [0x37] = DEVINFO_RT, [0x38] = DEVINFO_RT, [0x39] = 128,
    [0x3A] = DEVINFO_RT, [0x3B] = 1, [0x3C] = 4, [0x3D] = DEVINFO_RT, [0x3E] = DEVINFO_RT, [0x3F] = DEVINFO_RT, [0x40] = DEVINFO_RT, [0x41] = DEVINFO_RT, [0x42] = 1, [0x43] = 1,
    [0x44] = 1, [0x45] = 1, [0x46] = 5, [0x47] = 1, [0x48] = 1, [0x49] = 1, [0x4A] = 1, [0x4B] = 1, [0x4C] = 1, [0x4D] = 16, [0x4E] = 2048, [0x4F] = 32, [0x50] = 4096,
    [0x51] = 65536, [0x52] = 65536, [0x53] = 256, [0x54] = 16384, [0x55] = 262144, [0x56] = 4096, [0x57] = 1, [0x58] = 4, [0x59] = 2, [0x5A] = 0, [0x5B] = 4096, [0x5C] = 16384,
    [0x5D] = 4096, [0x5E] = 128, [0x5F] = 32, [0x60] = 65535, [0x61] = 1, [0x62] = DEVINFO_RT, [0x63] = 256, [0x64] = 16, [0x65] = 65535, [0x66] = 65535, [0x67] = 256,
    [0x68] = 1, [0x69] = 16, [0x6A] = 16, [0x6B] = 16, [0x6C] = 1, [0x6D] = 1, [0x6E] = 1, [0x6F] = 256, [0x70] = 0, [0x71] = 16, [0x72] = 8, [0x73] = 16, [0x74] = 32,
    [0x75] = 31, [0x76] = 536870912, [0x77] = 16777216, [0x78] = 16777216, [0x79] = 1073741824, [0x7A] = 32, [0x7B] = 1, [0x7C] = 64, [0x7D] = DEVINFO_RT, [0x7E] = 1,
    [0x7F] = 2147483647, [0x80] = 32, [0x81] = 32, [0x82] = 4194303, [0x83] = 512, [0x84] = 256, [0x85] = 32, [0x86] = 16383, [0x87] = DEVINFO_RT, [0x88] = 8, [0x89] = 0,
    [0x8A] = 0, [0x8B] = 4, [0x8C] = 256, [0x8D] = 16, [0x8E] = 0, [0x8F] = 1, [0x90] = 31, [0x91] = 0, [0x92] = 64, [0x93] = 1, [0x94] = 58368, [0x95] = 24576, [0x96] = 0,
    [0x97] = 256, [0x98] = 262144, [0x99] = DEVINFO_RT, [0x9A] = DEVINFO_RT, [0x9B] = 0, [0x9C] = 201, [0x9D] = 14, [0x9E] = 201, [0x9F] = 20, [0xA0] = 163840, [0xA1] = 8,
    [0xA2] = 4, [0xA3] = 256, [0xA4] = 256, [0xA5] = 4, [0xA6] = 65536, [0xA7] = 201, [0xA8] = 5, [0xA9] = 1,
};
// The NVN API version the game's own driver would report follows its SDK: games check it and create no device on a mismatch. SDK 10 and older are API 53 (Super Mario 3D World 10.4,
// Age of Calamity 9.6, Link's Awakening 8, Odyssey 7.3, Three Houses 6.5); SDK 11 and later take 55.15
__attribute__((weak)) int dm_sdk_major(void) { return -1; }   // loader.c; graphics unit tests have no loader. Lichii: -1 unknown, 0 a prerelease SDK 0.x
__attribute__((weak)) int dm_sdk_ounce(void) { return 0; }
__attribute__((weak)) int dm_sdk_nvn2_item(unsigned item, int *out) { (void)item; (void)out; return 0; }
__attribute__((weak)) int dm_glslc_gpu_code(int *major, int *minor) { *major = *minor = 0; return 0; }   // hle_glslc.c: the game's own GLSLC's GPU code version
static void nvn_api(int api[2]) {
#ifdef DM_GAME_NVN_API_MAJOR
    api[0] = DM_GAME_NVN_API_MAJOR; api[1] = DM_GAME_NVN_API_MINOR; LOG("[nvn] device API %d.%d (game profile device_api)", api[0], api[1]);
#else
    // The Switch 2's NVN2 is API 231, and its minor moves with the SDK: the game's sdk holds the answer of its own driver (dm_sdk_nvn2_item). Games require it and create their graphics system
    // only then: Donkey Kong Bananza (SDK 20.5.19) major == 231 and minor >= 30 (main+0x5c61a4), Splatoon Raiders (SDK 21.4.4) minor >= 39 (main+0x31b86b0)
    int m = dm_sdk_major();
    if (dm_sdk_ounce()) { api[0] = 231; if (!dm_sdk_nvn2_item(1, &api[1])) api[1] = g_devinfo2[1]; }
    else if (m == 0) { api[0] = 52; api[1] = 32; }   // Lichii: prerelease SDK 0.12.14 (Breath of the Wild 1.6.0), measured from its own driver
    else { api[0] = DM_SDK_OLD(m) ? 53 : g_devinfo[0]; api[1] = DM_SDK_OLD(m) ? 313 : g_devinfo[1]; }
    LOG("[nvn] device API %d.%d (SDK %d%s)", api[0], api[1], m, dm_sdk_ounce() ? ", Switch 2" : "");
#endif
}
NVN(nvnDeviceGetInteger, void, const NVNdevice *d, NVNdeviceInfo what, int *out) {
    (void)d;
    unsigned w = (unsigned)what;
    // 0x5F separate sampler binding count and 0x60/0x61 are not in the table measured on SMO: 0x5F is given 4096 per the sampler pool limit, unverified; 0x60/0x61 report a failure and return 0
    // 0x67 (SDK 19): FINAL FANTASY TACTICS - The Ivalice Chronicles reads it before every program it loads and passes it as the alignment of its copies of the shader's control section and code
    // (main+0x49800; its allocator raises anything below 8 to 8). Not measured on hardware: 256, the alignment of shader code in a memory pool; the host reads shaders by key, so any value works
    if (dm_sdk_ounce() && w < sizeof g_devinfo2 / sizeof *g_devinfo2 && g_devinfo2[w] == DEVINFO_RT && dm_sdk_nvn2_item(w, out)) { OLOG("DeviceGetInteger %#x -> %d (the NVN2 driver's)", w, *out); return; }   // Lichii: captured from the real driver, below 0x60 too
    if (dm_sdk_ounce() && w < sizeof g_devinfo2 / sizeof *g_devinfo2 && (g_devinfo2[w] != DEVINFO_RT || w >= sizeof g_devinfo / sizeof *g_devinfo)) {
        if (g_devinfo2[w] == DEVINFO_RT) { LOG("[RED] nvnDeviceGetInteger item %#x: NVN2 reads it from the GPU at run time, not known here; returning 0", w); *out = 0; return; }
        *out = w <= 1 ? 0 : g_devinfo2[w];
        if (w > 1) { dm_sdk_nvn2_item(w, out); OLOG("DeviceGetInteger %#x -> %d (NVN2)", w, *out); return; }   // the sdk's own constant where its switch has one: the GPU code versions move with the SDK
    } else if (w == 0x67) { *out = 256; return; }
    if (w >= sizeof g_devinfo / sizeof *g_devinfo) { LOG("[RED] nvnDeviceGetInteger unknown item %#x, returning 0", w); *out = 0; return; }
    *out = g_devinfo[w];
    // 0x2E / 0x30 describe the GPU code the host can translate, not the linked SDK's driver. Precompiled shaders can come from a newer compiler: ASTRAL CHAIN links SDK 6.4 but its
    // hwfillcolor.vso is 1.15; main+0x9e97d0 rejects the old SDK-based answer 1.14 and initialization unwinds into a null read at main+0x9c0d1c. Our chain translates through 1.16
    // (Bayonetta 3), so advertise that range for every SDK. Retain the floor from a game's own run-time GLSLC (Catherine: Full Body), whose outputs the build also translates.
    if (w == 0x2E || w == 0x30) {
        int gv[2], cur[2] = {1, 16};
        if (dm_glslc_gpu_code(&gv[0], &gv[1]) && (gv[0] > cur[0] || (gv[0] == cur[0] && gv[1] > cur[1]))) { cur[0] = gv[0]; cur[1] = gv[1]; }
        *out = w == 0x2E ? cur[0] : cur[1];
    }
    if (w <= 1) { pthread_once(&g_api_once, api_find); *out = g_api[w]; }   // once for every thread: a second thread reading api[] half written answered minor -1 (Mario Kart World then makes no graphics system)
    OLOG("DeviceGetInteger %#x -> %d", w, *out);
}
NVN(nvnDeviceGetImageHandle, NVNimageHandle, const NVNdevice *d, int id) { (void)d; return (NVNimageHandle)id; }
// Separate texture / sampler handles: value = descriptor index (the sampler index goes in bit 20 and above, consistent with the bit layout of combined handles)
NVN(nvnDeviceGetSeparateTextureHandle, NVNseparateTextureHandle, const NVNdevice *d, int id) { (void)d; return (NVNseparateTextureHandle){(uint64_t)id}; }
NVN(nvnDeviceGetSeparateSamplerHandle, NVNseparateSamplerHandle, const NVNdevice *d, int id) { (void)d; return (NVNseparateSamplerHandle){(uint64_t)id << 20}; }
NVN(nvnDeviceInstallDebugCallback, void, NVNdevice *d, const PFNNVNDEBUGCALLBACKPROC cb, void *data, uint32_t enable) {
    dev_t_ *o = (dev_t_ *)d; o->dbg_cb = B(enable) ? (void *)cb : NULL; o->dbg_data = data;
}
NVN(nvnDeviceSetWindowOriginMode, void, NVNdevice *d, NVNwindowOriginMode m) { ((dev_t_ *)d)->origin = m; mtl_set_origin_lower_left(m == NVN_WINDOW_ORIGIN_MODE_LOWER_LEFT); LOG("[nvn] window origin %s", m ? "upper left (as D3D / Metal)" : "lower left (GL)"); }
NVN(nvnDeviceSetDepthMode, void, NVNdevice *d, NVNdepthMode m) { ((dev_t_ *)d)->depth_mode = m; mtl_set_depth_zero(m != 0); LOG("[nvn] depth mode %s", m ? "near=0 (as D3D / Metal)" : "near=-w (GL)"); }
// Pokémon Sword sets neither mode and reads both, keeping NVN's defaults (lower-left origin, near = -w); the two were red stubs
NVN(nvnDeviceGetWindowOriginMode, NVNwindowOriginMode, const NVNdevice *d) { return ((const dev_t_ *)d)->origin; }
NVN(nvnDeviceGetDepthMode, NVNdepthMode, const NVNdevice *d) { return ((const dev_t_ *)d)->depth_mode; }
NVN(nvnDeviceFinalizeCommandHandle, void, NVNdevice *d, NVNcommandHandle h) { (void)d; (void)h; }

// ---------- Pool census (DM_NVN_POOLS=1) ----------
// What the game's memory pools hold, to tell which of them the GPU needs as its own memory (on Android every pool with storage becomes placed Vulkan memory, committed whole:
// docs/reference/memory.md "Placed pools commit what the game only reserved"). Per pool: its flags and size, the bytes its buffers and its textures cover (a union at 64 KiB),
// buffer-address queries, the bytes of it that physical mappings of virtual pools reach, and at the log the bytes that are not zero (4 KiB pages). Logged at presents 1, 60
// and 300, grouped by flags, then the largest pools. Coverage is what was ever covered (a finalized buffer or texture, an unmapped range keep their bits). Off: nothing is kept
enum { PC_SHIFT = 16 };
typedef struct { const pool_t *p; const uint8_t *mem; uint64_t size; uint32_t flags, naddr, nbuf, ntex, nmap; uint8_t *bb, *tb, *mb; } pcen;
static pcen *g_pc; static int g_npc, g_cpc, g_pc_on = -1;
static pthread_mutex_t g_pc_mu = PTHREAD_MUTEX_INITIALIZER;
static int pc_on(void) { if (g_pc_on < 0) g_pc_on = getenv("DM_NVN_POOLS") != NULL; return g_pc_on; }
static pcen *pc_find(const pool_t *p) { for (int i = g_npc - 1; i >= 0; i--) if (g_pc[i].p == p && g_pc[i].mem == p->mem) return &g_pc[i]; return NULL; }   // caller holds g_pc_mu
static void pc_add(const pool_t *p) {
    if (!pc_on() || !p->mem || p->flags & NVN_MEMORY_POOL_FLAGS_VIRTUAL) return;
    uint64_t nb = ((p->size + (1u << PC_SHIFT) - 1) >> PC_SHIFT) / 8 + 1;
    pthread_mutex_lock(&g_pc_mu);
    if (g_npc == g_cpc && !(g_pc = realloc(g_pc, (size_t)(g_cpc = g_cpc ? 2 * g_cpc : 256) * sizeof *g_pc))) die("pool census allocation failed");
    for (int i = 0; i < g_npc; i++) if (g_pc[i].p == p) { free(g_pc[i].bb); free(g_pc[i].tb); free(g_pc[i].mb); g_pc[i--] = g_pc[--g_npc]; }   // initialized again without a Finalize
    pcen c = {p, p->mem, p->size, (uint32_t)p->flags, 0, 0, 0, 0, calloc(nb, 1), calloc(nb, 1), calloc(nb, 1)};
    if (!c.bb || !c.tb || !c.mb) die("pool census allocation failed");
    g_pc[g_npc++] = c;
    pthread_mutex_unlock(&g_pc_mu);
}
static void pc_del(const pool_t *p) {
    if (!pc_on()) return;
    pthread_mutex_lock(&g_pc_mu);
    for (int i = 0; i < g_npc; i++) if (g_pc[i].p == p && g_pc[i].mem == p->mem) { free(g_pc[i].bb); free(g_pc[i].tb); free(g_pc[i].mb); g_pc[i] = g_pc[--g_npc]; break; }
    pthread_mutex_unlock(&g_pc_mu);
}
static void pc_mark(uint8_t *bm, uint64_t size, int64_t off, uint64_t n) {
    if (off < 0 || (uint64_t)off >= size || !n) return;
    uint64_t e = n > size - (uint64_t)off ? size : (uint64_t)off + n;
    for (uint64_t b = (uint64_t)off >> PC_SHIFT; b <= (e - 1) >> PC_SHIFT; b++) bm[b >> 3] |= (uint8_t)(1u << (b & 7));
}
static void pc_use(const pool_t *p, int kind, int64_t off, uint64_t n) {   // kind: 'b' buffer, 't' texture, 'a' address query, 'm' mapped into a virtual pool
    if (!pc_on() || !p) return;
    pthread_mutex_lock(&g_pc_mu);
    pcen *c = pc_find(p);
    if (c) switch (kind) {
        case 'b': c->nbuf++; pc_mark(c->bb, c->size, off, n); break;
        case 't': c->ntex++; pc_mark(c->tb, c->size, off, n); break;
        case 'm': c->nmap++; pc_mark(c->mb, c->size, off, n); break;
        default: c->naddr++;
    }
    pthread_mutex_unlock(&g_pc_mu);
}
static uint64_t pc_cover(const pcen *c, const uint8_t *bm, int none) {   // bytes of the 64 KiB blocks marked in bm, or (none) marked in no map
    uint64_t n = 0, nb = (c->size + (1u << PC_SHIFT) - 1) >> PC_SHIFT;
    for (uint64_t b = 0; b < nb; b++) {
        int m = none ? !((c->bb[b >> 3] | c->tb[b >> 3] | c->mb[b >> 3]) >> (b & 7) & 1) : bm[b >> 3] >> (b & 7) & 1;
        if (m) { uint64_t lo = b << PC_SHIFT, hi = lo + (1u << PC_SHIFT); n += (hi > c->size ? c->size : hi) - lo; }
    }
    return n;
}
static uint64_t pc_nonzero(const pcen *c) {   // bytes in 4 KiB pages that are not all zero (reads the whole pool)
    uint64_t n = 0;
    for (uint64_t o = 0; o < c->size; o += 4096) {
        uint64_t k = c->size - o < 4096 ? c->size - o : 4096, v = 0, q[512];
        if (dm_guest_read_mapped(q, (uintptr_t)(c->mem + o), k)) continue;   // (not mapped by the guest: inaccessible; copied, not read in place: the guest may unmap the page meanwhile)
        for (uint64_t i = 0; i < k / 8 && !v; i++) v = q[i];
        if (v) n += k;
    }
    return n;
}
static const pcen *g_pc_sort;
static int pc_cmp(const void *a, const void *b) { uint64_t x = g_pc_sort[*(const int *)a].size, y = g_pc_sort[*(const int *)b].size; return x < y ? 1 : x > y ? -1 : 0; }
static void pc_log(uint64_t present) {
    if (!pc_on()) return;
    enum { NG = 32, NTOP = 24 };
    struct { uint32_t flags; int n; uint64_t size, buf, tex, map, none, nz, naddr; } g[NG]; int ng = 0;
    pthread_mutex_lock(&g_pc_mu);
    uint64_t all = 0, allnz = 0;
    uint64_t *nz = calloc((size_t)g_npc + 1, sizeof *nz); int *ix = calloc((size_t)g_npc + 1, sizeof *ix);
    if (!nz || !ix) die("pool census allocation failed");
    for (int i = 0; i < g_npc; i++) {
        const pcen *c = &g_pc[i]; int k = 0; ix[i] = i;
        while (k < ng && g[k].flags != c->flags) k++;
        if (k == ng) { if (ng == NG) k = NG - 1; else { memset(&g[ng], 0, sizeof g[ng]); g[ng].flags = c->flags; ng++; } }
        nz[i] = pc_nonzero(c);
        g[k].n++; g[k].size += c->size; g[k].buf += pc_cover(c, c->bb, 0); g[k].tex += pc_cover(c, c->tb, 0); g[k].map += pc_cover(c, c->mb, 0);
        g[k].none += pc_cover(c, NULL, 1); g[k].nz += nz[i]; g[k].naddr += c->naddr; all += c->size; allnz += nz[i];
    }
    g_pc_sort = g_pc; qsort(ix, (size_t)g_npc, sizeof *ix, pc_cmp);
    LOG("[nvn] pools at present %llu: %d pools, %.0f MiB, %.0f MiB not zero (by flags: count, MiB, covered by buffers, by textures, mapped into virtual pools, by none, not zero, address queries)",
        (unsigned long long)present, g_npc, all / 1048576.0, allnz / 1048576.0);
    for (int k = 0; k < ng; k++)
        LOG("[nvn]   flags %#05x: %d, %.0f MiB, buffers %.0f, textures %.0f, mapped %.0f, none %.0f, not zero %.0f, address queries %llu", g[k].flags, g[k].n, g[k].size / 1048576.0,
            g[k].buf / 1048576.0, g[k].tex / 1048576.0, g[k].map / 1048576.0, g[k].none / 1048576.0, g[k].nz / 1048576.0, (unsigned long long)g[k].naddr);
    for (int t = 0; t < g_npc && t < NTOP; t++) {
        const pcen *c = &g_pc[ix[t]];
        LOG("[nvn]   pool %p+%#llx flags %#05x: %u buffers %.1f MiB, %u textures %.1f MiB, %u mappings %.1f MiB, none %.1f MiB, not zero %.1f MiB, %u address queries", (const void *)c->mem,
            (unsigned long long)c->size, c->flags, c->nbuf, pc_cover(c, c->bb, 0) / 1048576.0, c->ntex, pc_cover(c, c->tb, 0) / 1048576.0, c->nmap,
            pc_cover(c, c->mb, 0) / 1048576.0, pc_cover(c, NULL, 1) / 1048576.0, nz[ix[t]] / 1048576.0, c->naddr);
    }
    free(nz); free(ix);
    pthread_mutex_unlock(&g_pc_mu);
}

// ---------- Memory pools ----------
NVN(nvnMemoryPoolBuilderSetDefaults, void, NVNmemoryPoolBuilder *b) { memset(b, 0, sizeof(pool_builder_t)); }
NVN(nvnMemoryPoolBuilderSetDevice, void, NVNmemoryPoolBuilder *b, NVNdevice *d) { ((pool_builder_t *)b)->dev = d; }
NVN(nvnMemoryPoolBuilderSetStorage, void, NVNmemoryPoolBuilder *b, void *m, size_t n) { ((pool_builder_t *)b)->mem = m; ((pool_builder_t *)b)->size = n; }
NVN(nvnMemoryPoolBuilderSetFlags, void, NVNmemoryPoolBuilder *b, int f) { ((pool_builder_t *)b)->flags = f; }
// Lichii: a pool over a Switch 2 native image (Mario Kart World: GameChat's camera images, nx2:gsv 1211, each handle queried with gsv 428 first). The builder gets no storage, size or
// flags: the pool is the image's guest memory, as SetStorage would give it, CPU-uncached and GPU-cached unless flags were set.
NVN_NOPROTO(nvnMemoryPoolBuilderSetNativeHandleNVX, void, NVNmemoryPoolBuilder *b, uint64_t handle) {
    pool_builder_t *s = (pool_builder_t *)b; uint64_t address = 0, size = 0;
    if (!nvn_direct_native_image(handle, &address, &size)) { LOG("[RED] MemoryPoolBuilderSetNativeHandleNVX: native image %#llx is unknown or has no memory", (unsigned long long)handle); return; }
    s->mem = (void *)(uintptr_t)nvn_direct_host_of(address); s->size = size;
    if (!s->flags) s->flags = NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED | NVN_MEMORY_POOL_FLAGS_GPU_CACHED;
    LOG("MemoryPoolBuilderSetNativeHandleNVX: native image %llu at %#llx+%#llx", (unsigned long long)handle, (unsigned long long)address, (unsigned long long)size);
}
static uint8_t *vpool_reserve(pool_t *o); static void vpool_release(pool_t *o);   // sparse virtual pools, see MapVirtual
// Live pools, for threads that read a texture's pool after the game's call returned (the Vulkan replay). A registered texture keeps its pool's pointer, a guest address: the game
// may finalize the pool and unmap the memory holding the pool object (and the texture's storage) while the texture ID is still registered and not yet written by the replay
// (Mario Kart World on the phone, m22 / m23: SIGSEGV reading pool_t.vp of an unmapped pool object in nvn_sparse_pool, and earlier in an upload of that texture's storage). So the
// replay never reads a pool object directly: nvn_pool_live answers from this table, and reads the object only under its lock, which nvnMemoryPoolFinalize takes before anything is
// released (the game unmaps the object's memory after Finalize returns; vp itself changes under g_vp_mu, read here only as set or not). gen is the pool's Initialize serial: a
// pool initialized again at the same address is another pool. A reader that goes on to read the texture's storage holds nvn_pool_use (a read lock) from its liveness check until
// it is done; nvnMemoryPoolFinalize takes it for writing before the pool leaves the table, so a pool cannot be finalized, and its memory unmapped, under such a reader.
// Lichii: a guest unmap takes it for writing too, until the pages are gone (nvn_pool_unmap): a reader that found a page mapped reads it before the unmap, not after
typedef struct { const pool_t *p; uint64_t gen; } plive;
static plive *g_pl; static uint32_t g_npl, g_cpl; static uint64_t g_plgen;   // open addressing, g_cpl a power of two; p == (void *)1: a deleted slot; g_npl: used slots, deleted ones included
static pthread_mutex_t g_pl_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_rwlock_t g_pl_use = PTHREAD_RWLOCK_INITIALIZER;
static _Thread_local int t_pl_use;   // this thread's holds: only the outermost takes the lock (a waiting writer would block a second read lock)
void nvn_pool_use(int on) { if (on ? !t_pl_use++ : !--t_pl_use) { if (on) { if (pthread_rwlock_tryrdlock(&g_pl_use)) { void fritchii_hitch_add(int, unsigned long long); uint64_t t0 = now_ns(); pthread_rwlock_rdlock(&g_pl_use); fritchii_hitch_add(9, now_ns() - t0); } } else pthread_rwlock_unlock(&g_pl_use); } }
void nvn_pool_unmap(int on) { if (on) pthread_rwlock_wrlock(&g_pl_use); else pthread_rwlock_unlock(&g_pl_use); }
static uint32_t pl_hash(const pool_t *p) { uint64_t h = (uint64_t)(uintptr_t)p * 0x9E3779B97F4A7C15ull; return (uint32_t)(h >> 32); }
static _Atomic uint64_t g_plfin;   // pools finalized (or initialized again at a live pool's address) so far: nvn_pool_fin
uint64_t nvn_pool_fin(void) { return atomic_load_explicit(&g_plfin, memory_order_acquire); }
static plive *pl_find(const pool_t *p) {   // caller holds g_pl_mu
    if (!g_cpl) return NULL;
    for (uint32_t i = pl_hash(p) & (g_cpl - 1), k = 0; k < g_cpl; k++, i = (i + 1) & (g_cpl - 1)) { if (!g_pl[i].p) return NULL; if (g_pl[i].p == p) return &g_pl[i]; }
    return NULL;
}
static void pl_put(const pool_t *p, uint64_t gen) {   // caller holds g_pl_mu
    plive *e = pl_find(p); if (e) { e->gen = gen; atomic_fetch_add(&g_plfin, 1); return; }
    if ((g_npl + 1) * 2 > g_cpl) {   // rehash, dropping the deleted slots: twice the size only when the live pools need it
        uint32_t oc = g_cpl, live = 0; for (uint32_t i = 0; i < oc; i++) live += g_pl[i].p && g_pl[i].p != (const pool_t *)1;
        uint32_t nc = !oc ? 1024 : (live + 1) * 4 > oc ? oc * 2 : oc; plive *o = g_pl, *n = calloc(nc, sizeof *n); if (!n) die("live pool table allocation failed");
        g_pl = n; g_cpl = nc; g_npl = 0;
        for (uint32_t i = 0; i < oc; i++) if (o[i].p && o[i].p != (const pool_t *)1) pl_put(o[i].p, o[i].gen);
        free(o);
    }
    uint32_t i = pl_hash(p) & (g_cpl - 1); while (g_pl[i].p && g_pl[i].p != (const pool_t *)1) i = (i + 1) & (g_cpl - 1);
    if (!g_pl[i].p) g_npl++;   // (a deleted slot reused is counted already)
    g_pl[i] = (plive){p, gen};
}
void nvn_pool_note(const void *p) { pthread_mutex_lock(&g_pl_mu); pl_put((const pool_t *)p, ++g_plgen); pthread_mutex_unlock(&g_pl_mu); }   // a pool object the host made itself (the frame replayer)
static void pl_del(const pool_t *p) { pthread_mutex_lock(&g_pl_mu); plive *e = pl_find(p); if (e) { e->p = (const pool_t *)1; e->gen = 0; atomic_fetch_add(&g_plfin, 1); } pthread_mutex_unlock(&g_pl_mu); }   // (the slot stays counted until a grow)
// Whether pool is live: its Initialize serial (0: not live, or no pool), and under the table's lock its flags and whether it reads through a page table (nvn_sparse_pool)
uint64_t nvn_pool_live(const void *pool, unsigned *flags, int *sparse) {
    if (flags) *flags = 0;
    if (sparse) *sparse = 0;
    if (!pool) return 0;
    pthread_mutex_lock(&g_pl_mu); plive *e = pl_find((const pool_t *)pool); uint64_t g = e ? e->gen : 0;
    if (g) { const pool_t *o = (const pool_t *)pool; if (flags) *flags = (unsigned)o->flags; if (sparse) *sparse = o->vp && o->vmode != 1; }
    pthread_mutex_unlock(&g_pl_mu); return g;
}
NVN(nvnMemoryPoolInitialize, NVNboolean, NVNmemoryPool *p, const NVNmemoryPoolBuilder *b) {
    const pool_builder_t *s = (const pool_builder_t *)b; pool_t *o = (pool_t *)p;
    pthread_mutex_lock(&g_pl_mu);   // (a pool initialized again without a Finalize: readers see its new fields only with its new serial)
    o->magic = MAGIC('P', 'O', 'O', 'L'); o->flags = s->flags; o->mem = s->mem; o->size = s->size; o->vp = NULL; o->vmode = 0;
    pl_put(o, ++g_plgen); pthread_mutex_unlock(&g_pl_mu);
    if (!o->mem && (o->flags & 0x200)) { uint8_t *vr = vpool_reserve(o); if (vr) o->mem = vr; }   // a virtual pool's own address range, which its mappings fill (see MapVirtual)
    else if (o->mem) { mtl_pool_add(o->mem, o->size); nvn_dirty_add_range(o->mem, o->size); pc_add(o); }
    if (!o->mem && !(o->flags & 0x200)) LOG("[RED] MemoryPoolInitialize: non-virtual pool has no storage flags=%#x size=%#llx", o->flags, o->size);
    if (getenv("DM_NVN_VLOG")) { LOG("[nvn] MemoryPool %p mem=%p size=%#llx flags=%#x", (void *)p, (void *)o->mem, (unsigned long long)o->size, o->flags); void dm_backtrace(uint64_t fp); dm_backtrace((uint64_t)__builtin_frame_address(0)); }
    OLOG("MemoryPool %p mem=%p size=%#llx flags=%#x", (void *)p, (void *)o->mem, o->size, o->flags);
    return 1;
}
static void fl_add(const uint8_t *a, uint64_t n); static void fl_drop(const uint8_t *a, uint64_t n);   // flushed ranges, see nvn_flushed
static void hostseg_release(uintptr_t lo, uintptr_t hi);   // (command memory, below)
NVN(nvnMemoryPoolFinalize, void, NVNmemoryPool *p) { pthread_rwlock_wrlock(&g_pl_use); pl_del((const pool_t *)p); pthread_rwlock_unlock(&g_pl_use); pc_del((const pool_t *)p); if (((pool_t *)p)->flags & 0x200 && ((pool_t *)p)->vp) vpool_release((pool_t *)p);
    if (((pool_t *)p)->mem && !(((pool_t *)p)->flags & 0x200)) {   // A virtual pool's storage is borrowed from a physical pool; do not unregister it
 // Lichii: recordings in the pool's memory are dead with it, and so are the host-lent segments they continue in (the game's out-of-memory callback is never called
 // here, so a game that adds command memory only from it never gives that memory out again, and those segments were reclaimed by nothing else)
 hostseg_release((uintptr_t)((pool_t *)p)->mem, (uintptr_t)((pool_t *)p)->mem + ((pool_t *)p)->size);
 nvn_dirty_remove_range(((pool_t *)p)->mem); mtl_pool_del(((pool_t *)p)->mem); fl_drop(((pool_t *)p)->mem, ((pool_t *)p)->size); } mtl_alias_drop_pool(p); ((pool_t *)p)->magic = 0; }
static void ft_flush(const uint8_t *a, uint64_t n);
NVN(nvnMemoryPoolFlushMappedRange, void, const NVNmemoryPool *p, ptrdiff_t off, size_t n) {   // see ft_add and nvn_flushed
    const pool_t *o = (const pool_t *)p;
    if (!o->mem) return;
    if (!(o->flags & 0x200)) fl_add(o->mem + off, n);
    ft_flush(o->mem + off, n);
}
NVN(nvnMemoryPoolGetBufferAddress, NVNbufferAddress, const NVNmemoryPool *p) { pc_use((const pool_t *)p, 'a', 0, 0); if (((const pool_t *)p)->mem) mtl_pool_need(((const pool_t *)p)->mem, ((const pool_t *)p)->size, "the pool's address"); return (NVNbufferAddress)((const pool_t *)p)->mem; }

// ---------- Descriptor pools ----------
// One ID space per pool. In NVN a texture or sampler ID indexes the pool the command buffer has bound (SetTexturePool / SetSamplerPool), and two pools may use the same IDs. The replay keeps
// one table for each kind, so every pool gets a base in it, in creation order: registrations go to base + ID, and binds recorded on a command buffer get the base of the pool that command
// buffer last bound (bind_handles). A game with one pool of each kind is unchanged (base 0).
// Fire Emblem: Three Houses: a 12,544-entry texture pool and a 3,856-entry sampler pool for the scene, and 4,096-entry ones its 2D UI command buffer binds. Both fill IDs from 256 up; the scene's
// default 8×8 white texture overwrote the UI's textures, and the title menu's labels, the name entry and other menus drew as plain bars or not at all. The scene also registers each draw's
// textures into fresh IDs of a ring in its large pool (8,729 and up a few seconds in); with the table's capacity taken from the last pool alone, those all read as empty slots and the screen was black
enum { NDPOOL = 16 };
static struct dpool_base { const void *pool; int base, cap; } g_tpb[NDPOOL], g_spb[NDPOOL];   // cap: the entries its extent holds
static int g_ntpb, g_nspb, g_tnext, g_snext;
static pthread_mutex_t g_dpool_mu = PTHREAD_MUTEX_INITIALIZER;
static int dpool_add(struct dpool_base *t, int *nt, int *next, const void *pool, int n, const char *kind) {   // → the pool's base (assigned at its first Initialize, or a fresh one when it is initialized again larger)
    pthread_mutex_lock(&g_dpool_mu);
    int i = 0; while (i < *nt && t[i].pool != pool) i++;
    const int cap = (n + 255) & ~255;
    if (i == *nt) {
        if (*nt == NDPOOL) { pthread_mutex_unlock(&g_dpool_mu); LOG("[RED] %s pools exceed %d: later ones share the first pool's numbering", kind, NDPOOL); return 0; }
        t[i] = (struct dpool_base){pool, *next, cap}; (*nt)++; *next += cap;
        if (i) LOG("[nvn] pool #%d (%s, %d entries): numbered from %d in the host table, separate from the earlier pools", i + 1, kind, n, t[i].base);
    } else if (cap > t[i].cap) {   // its old extent would run into the next pool's
        t[i].base = *next; t[i].cap = cap; *next += cap;
        LOG("[nvn] pool #%d (%s) initialized again with %d entries: numbered from %d in the host table", i + 1, kind, n, t[i].base);
    }
    int b = t[i].base; pthread_mutex_unlock(&g_dpool_mu); return b;
}
static int dpool_base(const struct dpool_base *t, int nt, const void *pool) { for (int i = 0; i < nt; i++) if (t[i].pool == pool) return t[i].base; return 0; }
int nvn_dpool_base(const void *pool, int tex) { return tex ? dpool_base(g_tpb, g_ntpb, pool) : dpool_base(g_spb, g_nspb, pool); }   // the replay's bound pools (nvn_mtl.m state_op)
// The Android frame replayer (dm/gfx/vk_android/vka_replay.c), which runs in the host library without the game: a captured descriptor pool's bases, as its Initialize gave them
void nvn_dpool_base_set(const void *pool, int tex_base, int samp_base) {
    pthread_mutex_lock(&g_dpool_mu);
    if (g_ntpb < NDPOOL) g_tpb[g_ntpb++] = (struct dpool_base){pool, tex_base};
    if (g_nspb < NDPOOL) g_spb[g_nspb++] = (struct dpool_base){pool, samp_base};
    pthread_mutex_unlock(&g_dpool_mu);
}
// The pools each command buffer last bound are kept in its own host state (cmd_t tex_base / samp_base). They used to be a 64-entry list of command buffers that was never cleared:
// Dark Souls Remastered binds one of 12 sampler pools in more than 64 command buffers, and every later one read its samplers through the first pool's numbering
NVN(nvnTexturePoolInitialize, NVNboolean, NVNtexturePool *tp, const NVNmemoryPool *p, ptrdiff_t off, int n) {
    *(dpool_t *)tp = (dpool_t){(const pool_t *)p, off, n}; OLOG("TexturePool n=%d", n);
    { extern uint64_t g_tpool_lo, g_tpool_hi; const pool_t *pp = (const pool_t *)p; if (pp && pp->mem) { g_tpool_lo = (uint64_t)(uintptr_t)(pp->mem + off); g_tpool_hi = g_tpool_lo + (uint64_t)n * 32;
      LOG("[nvn] texture pool of %d descriptors at %#llx", n, (unsigned long long)g_tpool_lo); } }
    dpool_add(g_tpb, &g_ntpb, &g_tnext, tp, n, "texture"); mtl_set_tex_pool_n(g_tnext);
    return 1;
}
NVN(nvnTexturePoolFinalize, void, NVNtexturePool *tp) { (void)tp; }
// Fritchii: the descriptor's slot in the pool memory gets a tag naming the registration (the driver writes the texture's descriptor there): a game that copies
// descriptors between slots with the CPU (Ocarina of Time: its light table names slot 254, never registered) copies the tag, which the replay follows (vka_tex.c tag_id)
static void tpool_tag(const NVNtexturePool *tp, int id, int host_id) {
    const dpool_t *d = (const dpool_t *)tp;
    if (!d || !d->pool || !d->pool->mem || id < 0 || id >= d->n) return;
    uint32_t tag[2] = {0x58545246u /* "FRTX" */, (uint32_t)host_id};
    memcpy(d->pool->mem + d->off + (int64_t)id * 32, tag, sizeof tag);
}
NVN(nvnTexturePoolRegisterTexture, void, const NVNtexturePool *tp, int id, const NVNtexture *t, const NVNtextureView *v) {
    mtl_reg_tex(dpool_base(g_tpb, g_ntpb, tp) + id, (const tex_t *)t, (const view_t *)v);
    tpool_tag(tp, id, dpool_base(g_tpb, g_ntpb, tp) + id);
}
// Image descriptors live in the same pool index space as texture descriptors, and image handles are pool indices (nvnDeviceGetImageHandle; BindImages below), so an image
// registers into the same table. Hyrule Warriors: Age of Calamity registers the images its compute shaders write in the first battle (earlier games registered only textures)
NVN(nvnTexturePoolRegisterImage, void, const NVNtexturePool *tp, int id, const NVNtexture *t, const NVNtextureView *v) {
    mtl_reg_image(dpool_base(g_tpb, g_ntpb, tp) + id, (const tex_t *)t, (const view_t *)v);
    tpool_tag(tp, id, dpool_base(g_tpb, g_ntpb, tp) + id);
}
NVN(nvnSamplerPoolInitialize, NVNboolean, NVNsamplerPool *sp, const NVNmemoryPool *p, ptrdiff_t off, int n) {
    *(dpool_t *)sp = (dpool_t){(const pool_t *)p, off, n}; OLOG("SamplerPool n=%d", n); dpool_add(g_spb, &g_nspb, &g_snext, sp, n, "sampler"); return 1;
}
NVN(nvnSamplerPoolFinalize, void, NVNsamplerPool *sp) { (void)sp; }
NVN(nvnSamplerPoolRegisterSampler, void, const NVNsamplerPool *sp, int id, const NVNsampler *s) { mtl_reg_samp(dpool_base(g_spb, g_nspb, sp) + id, &((const samp_t *)s)->p); }
// straight from a builder, without a sampler object (FINAL FANTASY TACTICS - The Ivalice Chronicles registers its samplers this way)
NVN(nvnSamplerPoolRegisterSamplerBuilder, void, const NVNsamplerPool *sp, int id, const NVNsamplerBuilder *b) { mtl_reg_samp(dpool_base(g_spb, g_nspb, sp) + id, &((const samp_builder_t *)b)->p); }
// Unreal Engine reads a descriptor pool's size back at device setup (Octopath Traveler 0)
NVN(nvnTexturePoolGetMemoryPool, const NVNmemoryPool *, const NVNtexturePool *tp) { return (const NVNmemoryPool *)((const dpool_t *)tp)->pool; }
NVN(nvnTexturePoolGetMemoryOffset, ptrdiff_t, const NVNtexturePool *tp) { return (ptrdiff_t)((const dpool_t *)tp)->off; }
NVN(nvnTexturePoolGetSize, int, const NVNtexturePool *tp) { return ((const dpool_t *)tp)->n; }
NVN(nvnSamplerPoolGetMemoryPool, const NVNmemoryPool *, const NVNsamplerPool *sp) { return (const NVNmemoryPool *)((const dpool_t *)sp)->pool; }
NVN(nvnSamplerPoolGetMemoryOffset, ptrdiff_t, const NVNsamplerPool *sp) { return (ptrdiff_t)((const dpool_t *)sp)->off; }
NVN(nvnSamplerPoolGetSize, int, const NVNsamplerPool *sp) { return ((const dpool_t *)sp)->n; }

// ---------- Textures ----------
#define TB(b) ((tex_builder_t *)(b))
NVN(nvnTextureBuilderSetDefaults, void, NVNtextureBuilder *b) {
    NVNdevice *dev = TB(b)->dev; memset(b, 0, sizeof(tex_builder_t)); TB(b)->dev = dev;
    TB(b)->d.target = NVN_TEXTURE_TARGET_2D; TB(b)->d.width = TB(b)->d.height = TB(b)->d.depth = TB(b)->d.levels = 1;
}
NVN(nvnTextureBuilderSetDevice, void, NVNtextureBuilder *b, NVNdevice *d) { TB(b)->dev = d; }
NVN(nvnTextureBuilderSetFlags, void, NVNtextureBuilder *b, int f) { TB(b)->d.flags = f; }
NVN(nvnTextureBuilderSetTarget, void, NVNtextureBuilder *b, NVNtextureTarget t) { TB(b)->d.target = t; }
NVN(nvnTextureBuilderSetWidth, void, NVNtextureBuilder *b, int v) { TB(b)->d.width = v; }
NVN(nvnTextureBuilderSetHeight, void, NVNtextureBuilder *b, int v) { TB(b)->d.height = v; }
NVN(nvnTextureBuilderSetDepth, void, NVNtextureBuilder *b, int v) { TB(b)->d.depth = v; }
NVN(nvnTextureBuilderSetSize1D, void, NVNtextureBuilder *b, int w) { TB(b)->d.width = w; TB(b)->d.height = TB(b)->d.depth = 1; }
NVN(nvnTextureBuilderSetLevels, void, NVNtextureBuilder *b, int v) { TB(b)->d.levels = v; }
NVN(nvnTextureBuilderSetFormat, void, NVNtextureBuilder *b, NVNformat f) { TB(b)->d.format = f; }
NVN(nvnTextureBuilderSetSamples, void, NVNtextureBuilder *b, int s) { TB(b)->d.samples = s; }
NVN(nvnTextureBuilderSetStride, void, NVNtextureBuilder *b, ptrdiff_t s) { TB(b)->d.stride = (uint32_t)s; }
NVN(nvnTextureBuilderSetStorage, void, NVNtextureBuilder *b, NVNmemoryPool *p, ptrdiff_t off) { TB(b)->pool = (pool_t *)p; TB(b)->off = off; }
NVN(nvnTextureBuilderSetPackagedTextureLayout, void, NVNtextureBuilder *b, const NVNpackagedTextureLayout *l) {
    TB(b)->has_layout = l != NULL; if (l) memcpy(TB(b)->layout, l->layout, 8);
    TB(b)->d.pkg_block_h = l ? l->layout[0] + 1 : 0;
}
NVN(nvnTextureBuilderGetFlags, NVNtextureFlags, const NVNtextureBuilder *b) { return (NVNtextureFlags)((const tex_builder_t *)b)->d.flags; }
static uint64_t g_layout_logged;
static void log_layout(const tex_builder_t *b, const nvn_texlayout *L) {
    if (b->has_layout && g_layout_logged++ < 16)
        LOG("[nvn] packaged texture %ux%ux%u levels %u format %#x target %d layout bytes %02x%02x%02x%02x%02x%02x%02x%02x → block height log2=%u (from layout byte 0) size %#llx",
            b->d.width, b->d.height, b->d.depth, b->d.levels, b->d.format, b->d.target, b->layout[0], b->layout[1], b->layout[2],
            b->layout[3], b->layout[4], b->layout[5], b->layout[6], b->layout[7], L->tileH, L->size);
}
NVN(nvnTextureBuilderGetStorageSize, size_t, const NVNtextureBuilder *b) {
    nvn_texlayout L; nvn_tex_layout(&((const tex_builder_t *)b)->d, &L);
    if (!L.bpb) LOG("[RED] TextureBuilderGetStorageSize unknown format %#x", ((const tex_builder_t *)b)->d.format);
    log_layout((const tex_builder_t *)b, &L);
    return L.size;
}
NVN(nvnTextureBuilderGetStorageAlignment, size_t, const NVNtextureBuilder *b) {
    nvn_texlayout L; nvn_tex_layout(&((const tex_builder_t *)b)->d, &L); return L.align;
}
// The tiled levels of a sparse texture and its tile size (nvn_tex_sparse_tiles). OCTOPATH TRAVELER (Unreal Engine 4 on SDK 3.5.1) sizes a sparse texture's mip tail as storage(all
// levels) − storage(numTiledLevels levels) and allocates it from its heap at boot (main+0x1f1a688); unimplemented, the count was stack garbage and it asked for the whole 1.75 GB texture twice
static void sparse_layout(const nvn_texdesc *d, NVNtextureSparseTileLayout *o) {
    uint32_t tw, th, td; memset(o, 0, sizeof *o);
    o->numTiledLevels = nvn_tex_sparse_tiles(d, &tw, &th, &td); o->tileWidth = (int)tw; o->tileHeight = (int)th; o->tileDepth = (int)td;
}
NVN(nvnTextureBuilderGetSparseTileLayout, void, const NVNtextureBuilder *b, NVNtextureSparseTileLayout *o) { sparse_layout(&((const tex_builder_t *)b)->d, o); }
#define TX(t) ((tex_t *)(t))
#define CTX(t) ((const tex_t *)(t))
// Textures the CPU fills in place (profile tex_flush_upload): Tears of the Kingdom streams its textures (TexToGo, already in the GPU's block-linear layout) straight into
// CPU-cached memory pools and publishes each with nvnMemoryPoolFlushMappedRange (9,436 calls at boot), without a packaged layout or WriteTexels. They stayed GPU-only
// textures with content 0, and the G-buffer's base colour came out black. Textures that can hold such data (not render targets, not linear, no packaged layout, in a pool
// the CPU can write) are registered by address range; a flush that covers part of one marks it CPU-written (mtl_tex_cpu_written: uploaded from guest memory, re-uploaded
// when its pages change). Off by default: in another game a flushed range could cover an image a shader writes, whose GPU content the upload would replace
typedef struct { const uint8_t *mem; uint64_t size; tex_t t; uint64_t stamp; const void *span; } flush_tex;   // stamp: dirty-page generation at registration (tex_dirty_upload); span: its tracked range (nvn_dirty_span), so a check needs no search over every pool
static flush_tex *g_ft; static size_t g_nft, g_cft; static pthread_mutex_t g_ft_mu = PTHREAD_MUTEX_INITIALIZER;
static void ft_add(const tex_t *t) {
    // Buffer textures are views onto the pool buffer the GPU reads directly (tex_mtl in nvn_mtl.m); an upload would push Age of Calamity's 1,048,576-texel R32F one through tex_desc, past Metal's 2D limit
    if (!DM_GAME_TEX_FLUSH_UPLOAD || !t->mem || !t->pool || t->has_layout || t->d.target == NVN_TEXTURE_TARGET_BUFFER || (t->pool->flags & NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS) ||
        (t->d.flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET | NVN_TEXTURE_FLAGS_COMPRESSIBLE))) return;
    pthread_mutex_lock(&g_ft_mu);
    if (g_nft == g_cft && !(g_ft = realloc(g_ft, (g_cft = g_cft ? 2 * g_cft : 1024) * sizeof *g_ft))) die("flush_tex allocation failed");
    uint64_t st = nvn_dirty_now(); if (DM_GAME_TEX_DIRTY_UPLOAD) nvn_dirty_protect(t->mem, t->L.size);   // see nvn_ft_written
    g_ft[g_nft++] = (flush_tex){t->mem, t->L.size, *t, st, nvn_dirty_span(t->mem, t->L.size)};
    pthread_mutex_unlock(&g_ft_mu);
}
static void ft_del(uint64_t serial) {
    if (!DM_GAME_TEX_FLUSH_UPLOAD || !serial) return;
    pthread_mutex_lock(&g_ft_mu);
    for (size_t i = 0; i < g_nft; i++) if (g_ft[i].t.serial == serial) { g_ft[i] = g_ft[--g_nft]; break; }
    pthread_mutex_unlock(&g_ft_mu);
}
// Profile tex_dirty_upload: Tears of the Kingdom fills some textures from the CPU after creating them and publishes them with no NVN flush at all (os::FlushDataCache runs
// natively; its 121-layer ASTC terrain material arrays, 42 MB each). They stayed GPU-only textures with content 0, which ASTC decodes as the error colour, and the cave came out magenta.
// ft_add write-protects a registered texture's pages, and on its first resolution (tex_mtl, submit thread) a write since then makes it CPU-written like a flushed one
int nvn_ft_written(const tex_t *t) {
    if (!DM_GAME_TEX_DIRTY_UPLOAD || !t->serial) return 0;
    int r = 0; pthread_mutex_lock(&g_ft_mu);
    for (size_t i = 0; i < g_nft; i++) if (g_ft[i].t.serial == t->serial) {
        if (nvn_dirty_since_span(g_ft[i].span, g_ft[i].mem, g_ft[i].size, g_ft[i].stamp)) { r = 1; g_ft[i] = g_ft[--g_nft]; }
        break; }
    pthread_mutex_unlock(&g_ft_mu);
    if (r) mtl_tex_cpu_written(t);
    return r;
}
int nvn_ft_tracked(uint64_t serial) {   // registered and not resolved yet: its storage holds whatever the CPU has written (nvn_tex_mirror_b2t / nvn_tex_mirror_t2t)
    if (!DM_GAME_TEX_FLUSH_UPLOAD || !serial) return 0;
    int r = 0; pthread_mutex_lock(&g_ft_mu);
    for (size_t i = 0; i < g_nft; i++) if (g_ft[i].t.serial == serial) { r = 1; break; }
    pthread_mutex_unlock(&g_ft_mu);
    return r;
}
// Later writes (tex_dirty_upload): a texture first resolved before the CPU filled it stays a GPU-only texture, and the first-use check above never runs for it again.
// Once per submission, when guest memory has changed since the last look, the registered textures whose pages were written since registration become CPU-written
// (tex_mtl_ in nvn_mtl.m then drops the GPU-only texture and uploads it). Tears of the Kingdom fills its 5120x5120 BC4 roughness atlas from the CPU after first binding it,
// as tiles stream in. A texture the GPU writes (render target, shader-written image) may share pages with data the CPU writes: it leaves the list without an upload, which would
// replace the GPU's content
// Each poll looks at no more than FT_POLL_BYTES of registered textures, going on where the last one stopped; any guest write starts another pass over the whole list.
// Checking every texture at every submission cost TOTK 0.4 ms of submit CPU per frame on the Mac; this way a texture is picked up a few submissions later at most
enum { FT_POLL_BYTES = 128 << 20 };
void nvn_ft_poll(void) {
    if (!DM_GAME_TEX_DIRTY_UPLOAD) return;
    static uint64_t gen; static size_t cur, left;
    static tex_t *hit; static size_t chit; size_t nhit = 0;
    uint64_t now = nvn_dirty_now(), scanned = 0;
    pthread_mutex_lock(&g_ft_mu);
    if (now != gen) { gen = now; left = g_nft; }   // guest memory changed: every entry gets one more look
    while (left && g_nft && scanned < FT_POLL_BYTES) {
        if (cur >= g_nft) cur = 0;
        scanned += g_ft[cur].size; left--;
        if (nvn_dirty_since_span(g_ft[cur].span, g_ft[cur].mem, g_ft[cur].size, g_ft[cur].stamp)) {
            if (nhit == chit && !(hit = realloc(hit, (chit = chit ? 2 * chit : 64) * sizeof *hit))) die("nvn_ft_poll allocation failed");
            hit[nhit++] = g_ft[cur].t; g_ft[cur] = g_ft[--g_nft];
        } else cur++;
    }
    pthread_mutex_unlock(&g_ft_mu);
    for (size_t i = 0; i < nhit; i++) if (!mtl_tex_gpu_owned(hit[i].serial)) mtl_tex_cpu_written(&hit[i]);   // outside g_ft_mu: the ownership check takes the replay's serial table lock
}
static void ft_flush(const uint8_t *a, uint64_t n) {
    if (!DM_GAME_TEX_FLUSH_UPLOAD || !a || !n) return;
    pthread_mutex_lock(&g_ft_mu);
    for (size_t i = 0; i < g_nft;)
        if (g_ft[i].mem < a + n && a < g_ft[i].mem + g_ft[i].size) { mtl_tex_cpu_written(&g_ft[i].t); g_ft[i] = g_ft[--g_nft]; } else i++;
    pthread_mutex_unlock(&g_ft_mu);
}
// Flushed ranges (every game): what the CPU published with nvnMemoryPoolFlushMappedRange, as disjoint [lo, hi) spans sorted by start (touching spans merge), dropped with their pool.
// Until the GPU first uses a texture it has not written it, so storage the game flushed before that first use is the texture's content: tex_init_fill (nvn_mtl.m) fills the new GPU
// texture from it once, as it does for a pool the CPU cannot access. Breath of the Wild copies its precomputed sky scattering tables (System/KSys/sky.skybin: a 256x32x16 RGBA16F 3D
// texture and two 2D ones) into CPU_CACHED pools of their own with memcpy and flushes them (main+0x1257d50); as GPU textures of zeros the sky had no scattered light and came out
// brownish grey instead of blue (GitHub issue #52). Textures the CPU keeps writing after their first use still need the profile's tex_flush_upload / tex_dirty_upload
typedef struct { const uint8_t *lo, *hi; } fl_span;
static fl_span *g_fl; static size_t g_nfl, g_cfl; static pthread_mutex_t g_fl_mu = PTHREAD_MUTEX_INITIALIZER;
static size_t fl_find(const uint8_t *a, int touch) {   // the first span that ends after a (touch: or at a); g_fl_mu held
    size_t l = 0, h = g_nfl;
    while (l < h) { size_t m = (l + h) / 2; if (g_fl[m].hi < a || (!touch && g_fl[m].hi == a)) l = m + 1; else h = m; }
    return l;
}
static void fl_add(const uint8_t *a, uint64_t n) {
    if (!a || !n) return;
    const uint8_t *b = a + n;
    pthread_mutex_lock(&g_fl_mu);
    size_t i = fl_find(a, 1), j = i;
    if (i < g_nfl && g_fl[i].lo <= a && b <= g_fl[i].hi) { pthread_mutex_unlock(&g_fl_mu); return; }   // already covered (a buffer flushed every frame)
    while (j < g_nfl && g_fl[j].lo <= b) { if (g_fl[j].lo < a) a = g_fl[j].lo; if (g_fl[j].hi > b) b = g_fl[j].hi; j++; }   // spans i .. j-1 overlap or touch [a, b): merged into one
    if (i == j) {
        if (g_nfl == g_cfl && !(g_fl = realloc(g_fl, (g_cfl = g_cfl ? 2 * g_cfl : 256) * sizeof *g_fl))) die("flushed range allocation failed");
        memmove(g_fl + i + 1, g_fl + i, (g_nfl - i) * sizeof *g_fl); g_nfl++;
    } else if (j > i + 1) { memmove(g_fl + i + 1, g_fl + j, (g_nfl - j) * sizeof *g_fl); g_nfl -= j - i - 1; }
    g_fl[i] = (fl_span){a, b};
    pthread_mutex_unlock(&g_fl_mu);
}
static void fl_drop(const uint8_t *a, uint64_t n) {   // a pool goes: its part of every span (spans of neighbouring pools may have merged with it)
    const uint8_t *b = a + n;
    pthread_mutex_lock(&g_fl_mu);
    for (size_t i = fl_find(a, 0); i < g_nfl && g_fl[i].lo < b;) {
        fl_span s = g_fl[i];
        if (s.lo < a && s.hi > b) {   // the pool lies inside the span: split it
            if (g_nfl == g_cfl && !(g_fl = realloc(g_fl, (g_cfl = 2 * g_cfl) * sizeof *g_fl))) die("flushed range allocation failed");
            memmove(g_fl + i + 2, g_fl + i + 1, (g_nfl - i - 1) * sizeof *g_fl); g_nfl++;
            g_fl[i].hi = a; g_fl[i + 1] = (fl_span){b, s.hi}; break;
        }
        if (s.lo < a) { g_fl[i].hi = a; i++; }
        else if (s.hi > b) { g_fl[i].lo = b; break; }
        else { memmove(g_fl + i, g_fl + i + 1, (g_nfl - i - 1) * sizeof *g_fl); g_nfl--; }
    }
    pthread_mutex_unlock(&g_fl_mu);
}
int nvn_flushed(const void *p, uint64_t n) {   // whether the CPU flushed any of [p, p + n)
    const uint8_t *a = p;
    pthread_mutex_lock(&g_fl_mu);
    size_t i = fl_find(a, 0); int r = i < g_nfl && g_fl[i].lo < a + n;
    pthread_mutex_unlock(&g_fl_mu);
    return r;
}
NVN(nvnTextureInitialize, NVNboolean, NVNtexture *t, const NVNtextureBuilder *b) {
    const tex_builder_t *s = (const tex_builder_t *)b; tex_t *o = TX(t);
    // Re-initializing a texture object that is still live (without a Finalize first): release the host texture of the old serial first. SMBW re-initializes its transient render targets on the same object every frame and never Finalizes them;
    //   previously every init created a new private MTLTexture and the old one stayed in the serial table forever, so GPU memory grew about 13 MB per frame (killed by the memory watchdog at 24 GB)
    if (o->magic == MAGIC('T', 'E', 'X', 0) && o->serial) { static uint64_t n; if (!n++) LOG("[nvn] TextureInitialize on a texture object that is still live (no Finalize): releasing the old host texture (first case ser=%llu)", (unsigned long long)o->serial); mtl_tex_release(o); }
    memset(o, 0, sizeof *o);
    o->magic = MAGIC('T', 'E', 'X', 0); o->d = s->d; o->pool = s->pool; o->off = s->off;
    o->has_layout = s->has_layout; memcpy(o->layout, s->layout, 8); o->set_swz = s->set_swz; o->stencil = s->stencil; o->swz = s->swz;
    nvn_tex_layout(&o->d, &o->L);
    o->mem = s->pool && s->pool->mem ? s->pool->mem + s->off : NULL;
    o->serial = nvn_new_serial(); mtl_tex_init(o);
    if (o->mem) pc_use(o->pool, 't', o->off, o->L.size);
    if (o->mem && o->d.flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET)) mtl_pool_need(o->mem, o->L.size, "a linear texture");   // its GPU content is copied back into the storage (vka_tex_writeback)
    {   // DM_NVN_TEXINFO=WxH also logs every texture of that size as it is initialized (render targets included): which ones share storage, and how their descriptions differ
        static int ti = -1; static uint32_t tw, th;
        if (ti < 0) { const char *e = getenv("DM_NVN_TEXINFO"); ti = e && sscanf(e, "%ux%u", &tw, &th) == 2; }
        if (ti && o->d.width == tw && o->d.height == th) { static uint64_t n; if (n++ < 200) LOG("[nvn] TEXINFO init ser=%llu mem=%p pool=%p(mem %p size %#llx flags %#x) off=%#llx target=%d format=%#x flags=%#x samples=%d %ux%ux%u levels=%u stride=%u size=%#llx",
            (unsigned long long)o->serial, (void *)o->mem, (void *)o->pool, o->pool ? (void *)o->pool->mem : NULL, o->pool ? (unsigned long long)o->pool->size : 0ull, o->pool ? o->pool->flags : 0,
            (unsigned long long)o->off, o->d.target, o->d.format, o->d.flags, o->d.samples, o->d.width, o->d.height, o->d.depth, o->d.levels, o->d.stride, (unsigned long long)o->L.size); }
    }
    // See SetPackagedTextureData. Linear textures are not re-uploaded as packaged: they are created on the pool buffer (tex_linear), whatever the pool's CPU access, and read what was copied in
    if (s->pkg_data && o->mem) {
        memcpy(o->mem, s->pkg_data, o->L.size);
        if (!(o->d.flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET))) mtl_tex_cpu_written(o);
    }
    OLOG("Texture %p %ux%ux%u lv%u fmt=%#x tgt=%d flags=%#x size=%#llx", (void *)t, o->d.width, o->d.height, o->d.depth,
         o->d.levels, o->d.format, o->d.target, o->d.flags, o->L.size);
    ft_add(o);
    return 1;
}
NVN(nvnTextureFinalize, void, NVNtexture *t) { ft_del(TX(t)->serial); mtl_tex_release(TX(t)); TX(t)->magic = 0; }
NVN(nvnTextureGetFlags, NVNtextureFlags, const NVNtexture *t) { return (NVNtextureFlags)CTX(t)->d.flags; }
NVN(nvnTextureGetWidth, int, const NVNtexture *t) { return (int)CTX(t)->d.width; }
NVN(nvnTextureGetHeight, int, const NVNtexture *t) { return (int)CTX(t)->d.height; }
NVN(nvnTextureGetDepth, int, const NVNtexture *t) { return (int)CTX(t)->d.depth; }
NVN(nvnTextureGetStorageSize, int, const NVNtexture *t) { return (int)CTX(t)->L.size; }
NVN(nvnTextureGetStride, ptrdiff_t, const NVNtexture *t) { return (ptrdiff_t)CTX(t)->d.stride; }   // the row stride of a linear texture as built (0 for block-linear ones)
// The queries below are used by Engage (Unity) at startup: it calls MemoryPoolMap on "the pool the texture / buffer lives in", and with a stub returning 0 it dereferenced a null pool and crashed
NVN(nvnTextureGetLevels, int, const NVNtexture *t) { return (int)CTX(t)->d.levels; }
NVN(nvnTextureGetSparseTileLayout, void, const NVNtexture *t, NVNtextureSparseTileLayout *o) { sparse_layout(&CTX(t)->d, o); }
NVN(nvnTextureGetSamples, int, const NVNtexture *t) { return CTX(t)->d.samples; }
NVN(nvnTextureGetMemoryPool, NVNmemoryPool *, const NVNtexture *t) { return (NVNmemoryPool *)CTX(t)->pool; }
NVN(nvnTextureGetMemoryOffset, ptrdiff_t, const NVNtexture *t) { return (ptrdiff_t)CTX(t)->off; }
// The GPU address of the texture's storage (DRAGON QUEST XI S, Unreal Engine 4.18, asks for it at start): pool address + offset, the same address space as nvnMemoryPoolGetBufferAddress
NVN(nvnTextureGetTextureAddress, NVNtextureAddress, const NVNtexture *t) { const tex_t *o = CTX(t); if (o->pool && o->pool->mem) mtl_pool_need(o->pool->mem + o->off, o->L.size, "a texture's address"); return o->pool ? (NVNtextureAddress)(o->pool->mem + o->off) : 0; }
// The size of a texture's ZCull save area (SaveZCullData / RestoreZCullData). The host does no ZCull and never touches the area, but games allocate it from their own pools, and NVN
// does not report 0. Cuphead (Unity 2017.4) allocates one per depth target (main+0x1948060) from its pool allocator, which gives a zero-size request the offset of a live block; its
// map of allocations then loses that block (the zero-size one's size overwrites it, and freeing the zero-size one deletes it), and the block's own free later pushes a garbage-sized
// range onto the free list (main+0x1945b10). Buffers allocated after that overlap live ones: copies read staging bytes other copies had overwritten, and sprites broke into slivers,
// went missing, or flickered every fourth frame (the upload ring's period). The hardware's formula is not known; this one grows with the texture: 8 bytes per 32x32 tile and layer,
// at least 256
static size_t zcull_size(const nvn_texdesc *d) {
    uint64_t tiles = (uint64_t)((d->width + 31) / 32) * ((d->height + 31) / 32) * (d->depth ? d->depth : 1);
    return tiles ? (size_t)((tiles * 8 + 0xff) & ~0xffull) : 0x100;
}
NVN(nvnTextureGetZCullStorageSize, size_t, const NVNtexture *t) { return zcull_size(&CTX(t)->d); }
static unsigned tex_swz(const tex_t *t) { return t->set_swz ? t->swz : 2u | 3u << 3 | 4u << 6 | 5u << 9; }   // the channel mapping: identity R G B A when TextureBuilderSetSwizzle was not called
// The mapping counts: BotW and Odyssey rebuild layout captures on the same storage with a new mapping, and replace the registered descriptor only if the two compare different (BotW main+0xec33f0)
NVN(nvnTextureCompare, NVNboolean, const NVNtexture *a, const NVNtexture *b) {
    const tex_t *x = CTX(a), *y = CTX(b);
    return !memcmp(&x->d, &y->d, sizeof x->d) && x->mem == y->mem && tex_swz(x) == tex_swz(y);
}
// No prototype in the header; call sites such as main 0x174899c compare the return value with 1 and set some status bit 0x100 if not equal. Meaning unverified; return 1 for now
NVN_NOPROTO(nvnTextureGetRawStorageClass, int, const NVNtexture *t) { (void)t; return 1; }
// The class a texture from this builder would report, so the two agree: EA SPORTS FC 27's Frostbite renderer asks the builder (the class names the page kind a virtual pool maps,
// NVNmappingRequest.storageClass, which the host's virtual pools do not distinguish)
NVN_NOPROTO(nvnTextureBuilderGetRawStorageClass, int, const NVNtextureBuilder *b) { (void)b; return 1; }
#define VW(v) ((view_t *)(v))
NVN(nvnTextureViewSetDefaults, void, NVNtextureView *v) { memset(v, 0, sizeof(view_t)); }
NVN(nvnTextureViewSetLevels, void, NVNtextureView *v, int base, int n) { VW(v)->set_levels = 1; VW(v)->base_level = base; VW(v)->nlevels = n; }
NVN(nvnTextureViewSetLayers, void, NVNtextureView *v, int min, int n) { VW(v)->set_layers = 1; VW(v)->min_layer = min; VW(v)->nlayers = n; }
NVN(nvnTextureViewSetFormat, void, NVNtextureView *v, NVNformat f) { VW(v)->set_format = 1; VW(v)->format = f; }
NVN(nvnTextureViewSetTarget, void, NVNtextureView *v, NVNtextureTarget t) { VW(v)->set_target = 1; VW(v)->target = t; }
NVN(nvnTextureViewGetLayers, NVNboolean, const NVNtextureView *v, int *min, int *n) {
    const view_t *o = (const view_t *)v; *min = o->min_layer; *n = o->nlayers; return o->set_layers;
}
NVN(nvnTextureGetViewOffset, ptrdiff_t, const NVNtexture *t, const NVNtextureView *v) {
    const tex_t *o = CTX(t); const view_t *w = (const view_t *)v;
    if (!w) return 0;
    uint64_t off = w->set_levels ? nvn_tex_level_offset(&o->d, (unsigned)w->base_level) : 0;
    if (w->set_layers) off += (uint64_t)w->min_layer * o->L.layer_size;
    return (ptrdiff_t)off;
}
// CPU texture read (NVN ReadTexels): for packaged textures the pool memory is the ground truth (de-layout and read directly); GPU-written textures are read back from the MTLTexture (nvn_mtl.m)
static void read_texels(const tex_t *o, const view_t *v, const NVNcopyRegion *r, uint8_t *dst, ptrdiff_t row, ptrdiff_t img) {
    int bw, bh, bpb = nvn_fmt_bpb(o->d.format, &bw, &bh);
    uint32_t wb = (uint32_t)((r->width + bw - 1) / bw) * bpb, hb = (uint32_t)(r->height + bh - 1) / bh;
    if (!row) row = wb;
    if (!img) img = row * hb;
    unsigned level = v && v->set_levels ? (unsigned)v->base_level : 0, layer0 = v && v->set_layers ? (unsigned)v->min_layer : 0;
    mtl_read_texels(o, level, layer0, r, dst, (size_t)row, (size_t)img);
}
NVN(nvnTextureReadTexels, void, const NVNtexture *t, const NVNtextureView *v, const NVNcopyRegion *r, void *dst) {
    read_texels(CTX(t), (const view_t *)v, r, dst, 0, 0);
}
NVN(nvnTextureReadTexelsStrided, void, const NVNtexture *t, const NVNtextureView *v, const NVNcopyRegion *r, void *dst, ptrdiff_t row, ptrdiff_t img) {
    read_texels(CTX(t), (const view_t *)v, r, dst, row, img);
}
NVN(nvnTextureInvalidateTexels, void, const NVNtexture *t, const NVNtextureView *v, const NVNcopyRegion *r) { (void)t; (void)v; (void)r; }

// ---------- Samplers ----------
#define SB(b) (&((samp_builder_t *)(b))->p)
NVN(nvnSamplerBuilderSetDefaults, void, NVNsamplerBuilder *b) {
    NVNdevice *dev = ((samp_builder_t *)b)->dev; memset(b, 0, sizeof(samp_builder_t)); ((samp_builder_t *)b)->dev = dev;
    samp_par *o = SB(b);
    o->minf = NVN_MIN_FILTER_NEAREST_MIPMAP_LINEAR; o->magf = NVN_MAG_FILTER_LINEAR;   // GL defaults
    o->ws = o->wt = o->wr = NVN_WRAP_MODE_REPEAT; o->cmp_func = NVN_COMPARE_FUNC_LESS;
    o->lod_min = -1000.f; o->lod_max = 1000.f; o->aniso = 1.f;
}
NVN(nvnSamplerBuilderSetDevice, void, NVNsamplerBuilder *b, NVNdevice *d) { ((samp_builder_t *)b)->dev = d; }
NVN(nvnSamplerBuilderSetMinMagFilter, void, NVNsamplerBuilder *b, NVNminFilter mi, NVNmagFilter ma) { SB(b)->minf = mi; SB(b)->magf = ma; }
NVN(nvnSamplerBuilderSetWrapMode, void, NVNsamplerBuilder *b, NVNwrapMode s, NVNwrapMode t, NVNwrapMode r) { SB(b)->ws = s; SB(b)->wt = t; SB(b)->wr = r; }
NVN(nvnSamplerBuilderSetLodClamp, void, NVNsamplerBuilder *b, float lo, float hi) { SB(b)->lod_min = lo; SB(b)->lod_max = hi; }
NVN(nvnSamplerBuilderSetLodBias, void, NVNsamplerBuilder *b, float v) { SB(b)->lod_bias = v; }
NVN(nvnSamplerBuilderSetCompare, void, NVNsamplerBuilder *b, NVNcompareMode m, NVNcompareFunc f) { SB(b)->cmp_mode = m; SB(b)->cmp_func = f; }
NVN(nvnSamplerBuilderSetBorderColor, void, NVNsamplerBuilder *b, const float *c) { memcpy(SB(b)->border, c, 16); SB(b)->border_int = 0; }
// Metal's fixed border colours are selected numerically, also for integer textures (samp_border). Bit-casting integer 1 to float would select transparent black instead of opaque black/white.
NVN(nvnSamplerBuilderSetBorderColori, void, NVNsamplerBuilder *b, const int *c) { memcpy(SB(b)->border, c, 16); SB(b)->border_int = 1; }
NVN(nvnSamplerBuilderSetBorderColorui, void, NVNsamplerBuilder *b, const uint32_t *c) { memcpy(SB(b)->border, c, 16); SB(b)->border_int = 2; }
NVN(nvnSamplerBuilderSetMaxAnisotropy, void, NVNsamplerBuilder *b, float v) { SB(b)->aniso = v; }
NVN(nvnSamplerBuilderSetLodSnap, void, NVNsamplerBuilder *b, float v) { ((samp_builder_t *)b)->lod_snap = v; }   // No Metal equivalent: kept for the getter
NVN(nvnSamplerBuilderGetMinMagFilter, void, const NVNsamplerBuilder *b, NVNminFilter *mi, NVNmagFilter *ma) {
    const samp_par *o = &((const samp_builder_t *)b)->p; *mi = (NVNminFilter)o->minf; *ma = (NVNmagFilter)o->magf;
}
NVN(nvnSamplerBuilderGetWrapMode, void, const NVNsamplerBuilder *b, NVNwrapMode *s, NVNwrapMode *t, NVNwrapMode *r) {
    const samp_par *o = &((const samp_builder_t *)b)->p; *s = (NVNwrapMode)o->ws; *t = (NVNwrapMode)o->wt; *r = (NVNwrapMode)o->wr;
}
NVN(nvnSamplerBuilderGetLodClamp, void, const NVNsamplerBuilder *b, float *lo, float *hi) {   // Bayonetta 3 reads it back while building its samplers
    const samp_par *o = &((const samp_builder_t *)b)->p; *lo = o->lod_min; *hi = o->lod_max;
}
NVN(nvnSamplerInitialize, NVNboolean, NVNsampler *s, const NVNsamplerBuilder *b) {
    samp_t *o = (samp_t *)s; o->magic = MAGIC('S', 'M', 'P', 0); o->p = ((const samp_builder_t *)b)->p; o->lod_snap = ((const samp_builder_t *)b)->lod_snap; return 1;
}
NVN(nvnSamplerFinalize, void, NVNsampler *s) { ((samp_t *)s)->magic = 0; }

// ---------- Programs ----------
NVN(nvnProgramInitialize, NVNboolean, NVNprogram *p, NVNdevice *d) {
    (void)d; memset(p, 0, sizeof(prog_t)); ((prog_t *)p)->magic = MAGIC('P', 'R', 'O', 'G'); return 1;
}
static void pk_drop(const void *p);
NVN(nvnProgramFinalize, void, NVNprogram *p) { mtl_prog_release((prog_t *)p); ((prog_t *)p)->magic = 0; pk_drop(p); }
// a program's per-stage keys (side table: guest NVNprogram address → host object; bit i of ok = stage i computed)
typedef struct { uint8_t key[6][8]; uint8_t ok; } prog_keys_t;
static pthread_mutex_t g_pk_mu = PTHREAD_MUTEX_INITIALIZER; static plat_ptrmap_t *g_pk;   // a dedicated table (keys not shared with other side tables)
static prog_keys_t *pk_get(const void *p, int make) {
    pthread_mutex_lock(&g_pk_mu);
    if (!g_pk) g_pk = plat_ptrmap_create();
    prog_keys_t *k = (prog_keys_t *)plat_ptrmap_get(g_pk, p);
    if (!k && make) { k = calloc(1, sizeof *k); plat_ptrmap_set(g_pk, p, k); }
    pthread_mutex_unlock(&g_pk_mu);
    return k;
}
static void pk_drop(const void *p) {
    pthread_mutex_lock(&g_pk_mu);
    prog_keys_t *k = g_pk ? (prog_keys_t *)plat_ptrmap_get(g_pk, p) : NULL;
    if (k) { plat_ptrmap_remove(g_pk, p); free(k); }
    pthread_mutex_unlock(&g_pk_mu);
}
// mapped = 1: computed only when the code area is in a registered memory pool (at SetShaders: Mario sets some programs with an unmapped address first and fills it in before the first draw)
// The control section's bytes in the key and the code area's length. NVN: +0x704 is the code area's length; +0x20 holds the section's own length: sections of at least
// 0x880 bytes (all of Mario's, Engage's, Dread's and 13 Sentinels') use the first 0x880; SDK 10's 3D World has 0x850-byte ones followed by other data, so only their own
// length is used (shader_split.py ctl_len). NVN2 (Switch 2, version 0xc9 at +4): the code area's length at +0x44; the key takes the first 0x48 bytes (through the shader's
// own hash at +0x20 and the code lengths), +0x30 hashed as zero: Donkey Kong Bananza's BNSH files hold the section's length there (0xd8, 0xd0 in effects), agl's BAHS 0, also in memory
#define NVN2_KEY_CTL 0x48
static int nvn2_ctl(const uint8_t *c) { uint32_t magic, ver; memcpy(&magic, c, 4); memcpy(&ver, c + 4, 4); return magic == 0x98761234u && ver == 0xc9; }
void nvn_ctl_lens(const void *control, uint32_t *nctl, uint32_t *ncode) {
    const uint8_t *c = control; uint32_t own;
    if (nvn2_ctl(c)) { memcpy(ncode, c + 0x44, 4); *nctl = NVN2_KEY_CTL; return; }
    memcpy(ncode, c + 0x704, 4);
    memcpy(&own, c + 0x20, 4);
    *nctl = own >= 0x760 && own < 0x880 ? own : 0x880;
}
static int keys_hash(const NVNshaderData *sd, int mapped, uint8_t out[8]) {
    uint32_t ncode = 0, nctl; void *mb; uint64_t mo;
    if (!sd->control) return 0;
    nvn_ctl_lens(sd->control, &nctl, &ncode);
    int code = sd->data && ncode && ncode < (1u << 24);
    if (mapped && !(code && mtl_addr(sd->data, ncode, &mb, &mo))) return 0;
    uint8_t md[32]; plat_sha256_t hc; plat_sha256_init(&hc);
    // Profile key shader_key_skip: bytes of the control section hashed as zeros (shader_split.py key_ctl). Batman: Arkham Knight's engine shares one control section among shaders whose
    // sections differ only in GLSLC's compile information (+0x778..+0x791, +0x7d0..+0x7e7): it hands a program the section of an earlier shader with the same rest, so the bytes there name another shader
    if (nvn2_ctl(sd->control)) {
        uint8_t ctl[NVN2_KEY_CTL]; memcpy(ctl, sd->control, NVN2_KEY_CTL); memset(ctl + 0x30, 0, 4);
        plat_sha256_update(&hc, ctl, NVN2_KEY_CTL);
    } else if (DM_GAME_SHADER_KEY_SKIP_LEN && DM_GAME_SHADER_KEY_SKIP_OFF + DM_GAME_SHADER_KEY_SKIP_LEN <= nctl) {
        uint8_t ctl[0x880]; memcpy(ctl, sd->control, nctl); memset(ctl + DM_GAME_SHADER_KEY_SKIP_OFF, 0, DM_GAME_SHADER_KEY_SKIP_LEN);
        plat_sha256_update(&hc, ctl, nctl);
    } else plat_sha256_update(&hc, sd->control, nctl);
    if (code) plat_sha256_update(&hc, (const void *)(uintptr_t)sd->data, ncode);
    plat_sha256_final(md, &hc); memcpy(out, md, 8);
    return 1;
}
int prog_key(const prog_t *p, unsigned st, uint8_t key[8]) {
    if (st >= p->nstage) return 0;
    prog_keys_t *k = pk_get(p, 1);
    if (!(k->ok >> st & 1)) { keys_hash(&p->sd[st], 0, k->key[st]); k->ok |= (uint8_t)(1u << st); }
    memcpy(key, k->key[st], 8);
    return 1;
}
// The Android frame replayer (vka_replay.c): a captured program's stage keys (its shader data is not in the capture)
void nvn_prog_keys_set(const prog_t *p, const uint8_t key[6][8], uint8_t ok) { prog_keys_t *k = pk_get(p, 1); memcpy(k->key, key, sizeof k->key); k->ok = ok; }
NVN(nvnProgramSetShaders, NVNboolean, NVNprogram *p, int n, const NVNshaderData *sd) {
    prog_t *o = (prog_t *)p;
    if (n < 0 || n > 6) { LOG("[RED] ProgramSetShaders stage count %d", n); return 0; }
    mtl_prog_release(o); o->nstage = n; memcpy(o->sd, sd, n * sizeof *sd); o->serial = nvn_new_serial();
    // The keys are computed here (see prog_key in nvn_obj.h): the control section is only guaranteed valid during this call; the code area lives in a memory pool and stays valid while the program is in use
    static int dump = -1; if (dump < 0) dump = getenv("DM_NVN_SHDUMP") != NULL;
    static const char *trace; static int trace_init;
    if (!trace_init) { trace = getenv("DM_NVN_PROGTRACE"); trace_init = 1; }
    prog_keys_t *k = pk_get(p, 1);
    k->ok = 0;
    for (int i = 0; i < n; i++) {
        if (keys_hash(&sd[i], 1, k->key[i])) k->ok |= (uint8_t)(1u << i);
        // Diagnostic: trace the loader of selected shader keys (comma-separated hex).
        if (trace && (k->ok & (1u << i))) {
            char key[17]; for (int j = 0; j < 8; j++) snprintf(key + j * 2, 3, "%02x", k->key[i][j]);
            if (strstr(trace, key)) {
                LOG("[nvn] ProgramSetShaders key=%s control=%p code=%#llx", key, sd[i].control, (unsigned long long)sd[i].data);
                dm_backtrace((uint64_t)__builtin_frame_address(0));
            }
        }
        // DM_NVN_SHDUMP (debugging): keep a copy of the control section so the replay layer can save shaders missing from the map (never freed)
        // (an NVN2 section is 0xd8 bytes: only those are read, the rest of the copy is zero; a Maxwell one is 0x880 or more)
        if (dump && sd[i].control) { void *c = calloc(1, 0x880); if (c) { memcpy(c, sd[i].control, nvn2_ctl(sd[i].control) ? 0xd8 : 0x880); o->sd[i].control = c; } }
    }
    OLOG("ProgramSetShaders %p n=%d data0=%#llx", (void *)p, n, n ? (unsigned long long)sd[0].data : 0ull);
    mtl_prog_prefetch(o);   // its libraries load in the background now, not at its first draw
    return 1;
}

// ---------- Sync / events ----------
NVN(nvnSyncInitialize, NVNboolean, NVNsync *s, NVNdevice *d) { (void)d; *(sync_t *)s = (sync_t){MAGIC('S', 'Y', 'N', 'C'), 0, 0}; return 1; }
NVN(nvnSyncFinalize, void, NVNsync *s) { ((sync_t *)s)->magic = 0; }
NVN(nvnSyncSetDebugLabel, void, NVNsync *s, const char *l) { (void)s; (void)l; }
// Fence = "all commands previously submitted to the queue have completed": record the fence value of the most recent submit
// Lichii: the queue's own last submission, not the last of every queue: with the NVN queues as Vulkan queues another queue's submission may wait for an
// event this queue only signals after the game's wait for this sync (Mario Kart World hung waiting for its second queue's submission, which waited for the first's signal)
// Fritchii, FRITCHII_SYNC_LOG=<skip>,<count>: the game's queue syncs and submissions in call order, after <skip> of them
static int sync_log(void) { static long skip = -2, cnt; static _Atomic long n; if (skip == -2) { const char *e = getenv("FRITCHII_SYNC_LOG"); skip = -1; if (e) sscanf(e, "%ld,%ld", &skip, &cnt); }
    if (skip < 0) return 0; long k = atomic_fetch_add(&n, 1); return k >= skip && k < skip + cnt; }
NVN(nvnQueueFenceSync, void, NVNqueue *q, NVNsync *s, NVNsyncCondition c, int f) { (void)c; (void)f; ((sync_t *)s)->fenced = 1; ((sync_t *)s)->fence = ((const queue_t *)q)->last_fence;
    if (sync_log()) LOG("[nvn] sync log: FenceSync queue %p sync %p -> fence %llu", (void *)q, (void *)s, (unsigned long long)((const queue_t *)q)->last_fence); }
// A sync object that was never fenced waits until timeout, as on a real device; a fenced one is released only after the GPU has actually completed up to that fence value
// A GPU-side wait on a sync: the host has a single Metal queue that executes submissions in order, so commands before the fence are always done first and nothing needs waiting. A sync not fenced yet (fenced later by another thread) is not waited for here; logged once
// Lichii: the replay may run NVN queues on several Vulkan queues (V.multi), so a fence of QueueFenceSync goes with the queue's next submission, which waits for it on the GPU (mtl_sub_wait).
// A fence done already or kept already adds nothing. When NVN_QUEUE_WAITS fences not done yet are kept, this one is waited for here (at most the replay's hang time)
NVN(nvnQueueWaitSync, NVNboolean, NVNqueue *q, const NVNsync *s) {
    const sync_t *y = (const sync_t *)s; static int once;
    if (sync_log()) LOG("[nvn] sync log: WaitSync queue %p sync %p fenced %d fence %llu (done %d)", (void *)q, (void *)s, y->fenced, (unsigned long long)y->fence, y->fenced == 1 && y->fence ? mtl_wait(y->fence, 0) : -1);
    if (!y->fenced && !once++) LOG("[nvn] QueueWaitSync: the sync object has no fence yet, the host does not wait");
    if (y->fenced == 1 && y->fence && !mtl_wait(y->fence, 0)) {
        queue_t *o = (queue_t *)q; int kept = 0, full = 0; dm_host_mutex_lock(&g_sub_mu);
        for (int i = 0; i < o->nwait; i++) kept |= o->wait_fence[i] == y->fence;
        if (!kept && o->nwait == NVN_QUEUE_WAITS) { int n = 0; for (int i = 0; i < o->nwait; i++) if (!mtl_wait(o->wait_fence[i], 0)) o->wait_fence[n++] = o->wait_fence[i]; o->nwait = n; }
        if (!kept) { if (o->nwait < NVN_QUEUE_WAITS) o->wait_fence[o->nwait++] = y->fence; else full = 1; }
        pthread_mutex_unlock(&g_sub_mu);
        uint64_t vka_hang_ns(void);
        if (full && !mtl_wait(y->fence, vka_hang_ns())) { static int n; if (!n++) LOG("[RED] QueueWaitSync: more than %d fences to wait for before a submission, and fence %llu was not reached in time: not waited for", NVN_QUEUE_WAITS, (unsigned long long)y->fence); }
    }
    return 1;
}
// ZCull data: the host does no ZCull, so there is nothing to save / restore
NVN(nvnCommandBufferRestoreZCullData, void, NVNcommandBuffer *c, NVNbufferAddress a, size_t n) { (void)c; (void)a; (void)n; }
NVN(nvnCommandBufferSaveZCullData, void, NVNcommandBuffer *c, NVNbufferAddress a, size_t n) { (void)c; (void)a; (void)n; }
// The stencil test ZCull's stencil cull assumes (a stencil function, then its reference and mask): only a hint for culling ahead of the real stencil test, which the draws'
// depth-stencil state still does. The host does no ZCull, so there is nothing to set (Donkey Kong Bananza: EQUAL, 0xff)
NVN_NOPROTO(nvnCommandBufferSetStencilCullCriteria, void, NVNcommandBuffer *c, int func, int ref) { (void)c; (void)func; (void)ref; }
static _Atomic uint64_t g_sync_ns, g_sync_timeouts, g_finish_ns;
// Lichii, DM_NVN_WAIT_STATS=1: once a second, the game's blocked time in SyncWait and QueueFinish (the present calls it)
void nvn_wait_stats(uint64_t present_ns) {
    static int on = -1; if (on < 0) { const char *e = getenv("DM_NVN_WAIT_STATS"); on = e && atoi(e); }
    if (!on) return;
    static uint64_t t_last, frames, pres_ns, s0, f0, x0;
    uint64_t t = now_ns(); frames++; pres_ns += present_ns;
    if (!t_last) { t_last = t; return; }
    if (t - t_last < 1000000000ull) return;
    const double sec = (t - t_last) / 1e9; const uint64_t s = atomic_load(&g_sync_ns), f = atomic_load(&g_finish_ns), x = atomic_load(&g_sync_timeouts);
    LOG("[nvn] waits: %.1f frames/s; a second: SyncWait %.0f ms (%llu timeouts), QueueFinish %.0f ms, present call %.0f ms", frames / sec, (s - s0) / 1e6 / sec,
        (unsigned long long)(x - x0), (f - f0) / 1e6 / sec, pres_ns / 1e6 / sec);
    { uint64_t vka_frame(void); double vka_gpu_span_ms(uint64_t f), mtl_gpu_ms(uint64_t f); uint64_t fr = vka_frame(); double sp = 0, bu = 0; int n = 0;
      for (uint64_t k = fr - 40; k < fr - 10; k++) { double a = vka_gpu_span_ms(k), b = mtl_gpu_ms(k); if (a >= 0 && b >= 0) { sp += a; bu += b; n++; } }
      if (n) LOG("[nvn] GPU per frame: busy %.1f ms, span %.1f ms (%.1f waiting for the recording), %d frames", bu / n, sp / n, (sp - bu) / n, n); }
    static int every; if (++every % 10 == 0) { void mtl_pso_line(char *out, size_t n); char pl[1024]; mtl_pso_line(pl, sizeof pl); LOG("[nvn] pipelines: %s", pl); }
    t_last = t; frames = 0; pres_ns = 0; s0 = s; f0 = f; x0 = x;
}
#if defined(_WIN32)
__declspec(dllimport) unsigned long __stdcall GetCurrentThreadId(void);
static unsigned long nvn_sync_log_thread_id(void) { return (unsigned long)GetCurrentThreadId(); }
#else
#include <pthread.h>
static unsigned long nvn_sync_log_thread_id(void) { return (unsigned long)pthread_self(); }
#endif
NVN(nvnSyncWait, NVNsyncWaitResult, const NVNsync *s, uint64_t timeout) {
    const sync_t *o = (const sync_t *)s;
    if (sync_log()) { int pre = o->fenced && o->fence ? (o->fenced == 2 ? mtl_wait_acq : mtl_wait)(o->fence, 0) : -1; LOG("[nvn] sync log: SyncWait sync %p fenced %d fence %llu timeout %llu (done before %d) thread %lu", (void *)s, o->fenced, (unsigned long long)o->fence, (unsigned long long)timeout, pre, nvn_sync_log_thread_id()); }
    if (!o->fenced) return NVN_SYNC_WAIT_RESULT_TIMEOUT_EXPIRED;
    int (*w)(uint64_t, uint64_t) = o->fenced == 2 ? mtl_wait_acq : mtl_wait;
    if (w(o->fence, 0)) return NVN_SYNC_WAIT_RESULT_ALREADY_SIGNALED;
    uint64_t t0 = now_ns(); int ok = w(o->fence, timeout); atomic_fetch_add(&g_sync_ns, now_ns() - t0); { void fritchii_hitch_add(int, unsigned long long); fritchii_hitch_add(7, now_ns() - t0); }   // the frame's blocked SyncWait time (@sync_ms)
    if (!ok) atomic_fetch_add(&g_sync_timeouts, 1);
    return ok ? NVN_SYNC_WAIT_RESULT_CONDITION_SATISFIED : NVN_SYNC_WAIT_RESULT_TIMEOUT_EXPIRED;
}
NVN(nvnEventBuilderSetDefaults, void, NVNeventBuilder *b) { memset(b, 0, sizeof(ev_builder_t)); }
NVN(nvnEventBuilderSetStorage, void, NVNeventBuilder *b, const NVNmemoryPool *p, int64_t off) { *(ev_builder_t *)b = (ev_builder_t){(const pool_t *)p, off}; }
NVN(nvnEventInitialize, NVNboolean, NVNevent *e, const NVNeventBuilder *b) {
    const ev_builder_t *s = (const ev_builder_t *)b;
    if (!s->pool || !s->pool->mem) { LOG("[RED] EventInitialize has no storage"); return 0; }
    *(ev_t *)e = (ev_t){MAGIC('E', 'V', 'N', 'T'), (volatile uint32_t *)(s->pool->mem + s->off), s->pool, s->off}; return 1;
}
NVN(nvnEventFinalize, void, NVNevent *e) { ((ev_t *)e)->magic = 0; }
NVN(nvnEventGetValue, uint32_t, const NVNevent *e) { return *((const ev_t *)e)->val; }
static void ev_signal(const ev_t *e, int mode, uint32_t v) {
    volatile uint32_t *w = (volatile uint32_t *)nvn_direct_writable((uint64_t)(uintptr_t)e->val);   // Lichii: through the backing memory (nvn_tracking.c)
    uint32_t r = mode == NVN_EVENT_SIGNAL_MODE_ADD ? *w + v : v; *w = r;
#if defined(__ANDROID__)
    // (the replay's own order of signals and waits, vka_exec.c OP_WAIT_EVENT). Lichii: as the value written, which the replay's event caches take as is (an ADD they
    // would add again to the memory they seed from, which holds it already)
    mtl_event_cpu(e->val, NVN_EVENT_SIGNAL_MODE_WRITE, r);
#endif
}
NVN(nvnEventSignal, void, NVNevent *e, NVNeventSignalMode m, uint32_t v) { ev_signal((const ev_t *)e, m, v); }

// ---------- Queue ----------
#define QB(b) ((queue_builder_t *)(b))
NVN(nvnQueueBuilderSetDefaults, void, NVNqueueBuilder *b) {
    NVNdevice *dev = QB(b)->dev; memset(b, 0, sizeof(queue_builder_t)); QB(b)->dev = dev; QB(b)->prio = 100;
}
static void qb_set(NVNqueueBuilder *b, int k, size_t n) { QB(b)->sz[k] = (uint32_t)n; QB(b)->set[k] = 1; }
static uint64_t qb_size(const queue_builder_t *o, int k) { static const uint32_t def[4] = {65536, 262144, 16384, 4096}; return o->set[k] ? o->sz[k] : def[k]; }   // Device defaults for those not set
NVN(nvnQueueBuilderSetDevice, void, NVNqueueBuilder *b, NVNdevice *d) { QB(b)->dev = d; }
NVN(nvnQueueBuilderSetFlags, void, NVNqueueBuilder *b, int f) { QB(b)->flags = f; }
NVN(nvnQueueBuilderSetCommandMemorySize, void, NVNqueueBuilder *b, size_t n) { qb_set(b, QB_CMD, n); }
NVN(nvnQueueBuilderSetComputeMemorySize, void, NVNqueueBuilder *b, size_t n) { qb_set(b, QB_COMP, n); }
NVN(nvnQueueBuilderSetControlMemorySize, void, NVNqueueBuilder *b, size_t n) { qb_set(b, QB_CTL, n); }
NVN(nvnQueueBuilderSetCommandFlushThreshold, void, NVNqueueBuilder *b, size_t n) { qb_set(b, QB_FLUSH, n); }
// Queue memory: the sum of the command + compute + control blocks rounded to 4 KiB (the host does not use it; the game only needs to allocate this size)
NVN(nvnQueueBuilderGetQueueMemorySize, size_t, const NVNqueueBuilder *b) {
    const queue_builder_t *o = (const queue_builder_t *)b;
    return (qb_size(o, QB_CMD) + qb_size(o, QB_COMP) + qb_size(o, QB_CTL) + 4095) & ~(size_t)4095;
}
NVN(nvnQueueBuilderSetQueueMemory, void, NVNqueueBuilder *b, void *m, size_t n) { QB(b)->mem = m; QB(b)->mem_sz = (uint32_t)n; }
NVN(nvnQueueInitialize, NVNboolean, NVNqueue *q, const NVNqueueBuilder *b) {
    NVNdevice *d = ((const queue_builder_t *)b)->dev;
    // NVN emits SET_WINDOW_ORIGIN / FLIP_Y into the queue's initial GPU state.
    // Changing the device origin later changes viewport construction, but does
    // not rewrite that queue state (Xenoblade 2 creates its queue first).
    *(queue_t *)q = (queue_t){MAGIC('Q', 'U', 'E', 'U'), d, 0,
                            !d || ((const dev_t_ *)d)->origin == NVN_WINDOW_ORIGIN_MODE_LOWER_LEFT, nvn_new_serial()};
    OLOG("QueueInitialize %p", (void *)q); return 1;
}
NVN(nvnQueueFinalize, void, NVNqueue *q) {
    pthread_mutex_lock(&g_sub_mu); mtl_queue_forget(((queue_t *)q)->serial); ((queue_t *)q)->magic = 0; pthread_mutex_unlock(&g_sub_mu);
}
NVN(nvnQueueGetError, NVNqueueGetErrorResult, NVNqueue *q, NVNqueueErrorInfo *info) {
    (void)q; if (info) memset(info, 0, sizeof *info); return NVN_QUEUE_GET_ERROR_RESULT_GPU_NO_ERROR;
}
NVN(nvnQueueFlush, void, NVNqueue *q) { (void)q; }
NVN(nvnQueueFinish, void, NVNqueue *q) { uint64_t t0 = now_ns(); mtl_wait(((const queue_t *)q)->last_fence, UINT64_MAX); atomic_fetch_add(&g_finish_ns, now_ns() - t0); { void fritchii_hitch_add(int, unsigned long long); fritchii_hitch_add(7, now_ns() - t0); } }   // Lichii: the queue's own work (see nvnQueueFenceSync)

// ---------- Command buffer: record into the game's command memory, perform CPU-visible side effects at submit ----------
#define CB(c) ((cmd_t *)(c))
// Room kept at the segment end after every record but the END: one JUMP. The END needs none (nothing follows it), and every other record is at least 16 bytes, so a JUMP
// only ever stands where a record was to go: linking never writes past the bytes the recording takes in one piece, which is what GetCommandMemoryUsed reports and games size
// command memory by. Tears of the Kingdom (GitHub issue #61) gives each model's prerecorded lists fixed slots of the size a recording of that list measured, slot k at
// base + size × k, and passes AddCommandMemory max(size, 512) (main+0xbb120). With 24 bytes kept for "a JUMP or an END", a 496-byte list's END did not fit in 512,
// went to a host segment, and the 16-byte JUMP at 488 reached 8 bytes into the next slot; recording that slot wrote its first record over the JUMP's target, and the
// replay jumped to 0x1000000003 (that CALL record's header) after 8 to 30 minutes of play
#define OP_RESERVE 16

NVN(nvnCommandBufferInitialize, NVNboolean, NVNcommandBuffer *c, NVNdevice *d) {
    memset(c, 0, sizeof(cmd_t)); CB(c)->magic = MAGIC('C', 'M', 'D', 'B'); CB(c)->dev = d; return 1;
}
NVN(nvnCommandBufferFinalize, void, NVNcommandBuffer *c) { CB(c)->magic = 0; }
NVN(nvnCommandBufferSetMemoryCallback, void, NVNcommandBuffer *c, PFNNVNCOMMANDBUFFERMEMORYCALLBACKPROC cb) { CB(c)->cb = cb; }
NVN(nvnCommandBufferSetMemoryCallbackData, void, NVNcommandBuffer *c, void *d) { CB(c)->cb_data = d; }
static uint8_t *align8(uint8_t *p) { return (uint8_t *)(((uintptr_t)p + 7) & ~(uintptr_t)7); }
// Diagnostic DM_NVN_CMDSTAT=1: every 300 frames log one line of command recording usage -- how many bytes / records were recorded per frame, how large a command memory segment the game gives each time, how many out-of-memory callbacks per frame, how many host-supplied segments,
// and the count and bytes of each command type (to find "how much larger our records are than real-device commands"). Counts only; does not change recording / replay
static struct { int on; uint64_t bytes, recs, grants, grant_bytes, grant_min, grant_max, cbs, hostseg, op_n[64], op_b[64], frame0, waste; } g_cs = { .on = -1 };
static pthread_mutex_t g_cs_mu = PTHREAD_MUTEX_INITIALIZER;
uint64_t dm_nvn_frame_count(void);
static int cs_on(void) { if (g_cs.on < 0) g_cs.on = getenv("DM_NVN_CMDSTAT") != NULL; return g_cs.on; }
static void cs_op(uint32_t op, uint32_t size) { if (!cs_on() || op >= 64) return; pthread_mutex_lock(&g_cs_mu); g_cs.op_n[op]++; g_cs.op_b[op] += size; pthread_mutex_unlock(&g_cs_mu); }
static void cs_dump(void) {   // Called once per frame (at present)
    if (!cs_on()) return; uint64_t f = dm_nvn_frame_count(); if (f - g_cs.frame0 < 300) return;
    pthread_mutex_lock(&g_cs_mu); double n = (double)(f - g_cs.frame0); char top[700]; int k = 0; uint8_t used[64] = {0};
    for (int r = 0; r < 8; r++) { int bi = -1; for (int i = 0; i < 64; i++) if (!used[i] && g_cs.op_b[i] && (bi < 0 || g_cs.op_b[i] > g_cs.op_b[bi])) bi = i; if (bi < 0) break; used[bi] = 1;
        k += snprintf(top + k, sizeof top - k, " op %d: %.0f × %.0f bytes = %.1f KB;", bi, g_cs.op_n[bi] / n, (double)g_cs.op_b[bi] / (double)g_cs.op_n[bi], g_cs.op_b[bi] / n / 1024.0); }
    LOG("[nvn] command memory up to frame %llu: recorded per frame %.1f KB / %.0f records; segments from the game per frame %.2f, mean %.1f KB each (min %.1f / max %.1f); out-of-memory callbacks per frame %.2f; host-lent segments per frame %.3f; segment tails dropped on a switch per frame %.1f KB | per frame%s",
        (unsigned long long)f, g_cs.bytes / n / 1024.0, g_cs.recs / n, g_cs.grants / n, g_cs.grants ? g_cs.grant_bytes / (double)g_cs.grants / 1024.0 : 0, g_cs.grant_min / 1024.0, g_cs.grant_max / 1024.0, g_cs.cbs / n, g_cs.hostseg / n, g_cs.waste / n / 1024.0, top);
    int on = g_cs.on; memset(&g_cs, 0, sizeof g_cs); g_cs.on = on; g_cs.frame0 = f; pthread_mutex_unlock(&g_cs_mu);
}
static void put_hdr(uint8_t *p, uint32_t op, uint32_t size) { op_hdr h = {op, size}; memcpy(p, &h, sizeof h); if (op) cs_op(op, size); }
// New segment: if the recording in progress has records, write a JUMP at the end of the old segment to link it. A recording with none yet starts in the new segment
// (its first record is its handle), and the bytes at the old write position are not its own: after an END that took the last bytes there is no room there at all
static void cmd_new_segment(cmd_t *o, uint8_t *mem, size_t n) {
    uint8_t *start = align8(mem), *end = mem + n;
    if (o->recording == 1 && o->cur) { if (cs_on() && o->end > o->cur) { pthread_mutex_lock(&g_cs_mu); g_cs.waste += (uint64_t)(o->end - o->cur); pthread_mutex_unlock(&g_cs_mu); } put_hdr(o->cur, OP_JUMP, 16); memcpy(o->cur + 8, &start, 8);
        o->used += 16; }   // the JUMP also occupies guest memory of this recording: count it in used, so the game packing the next recording right behind by used does not overwrite it
    o->cur = start; o->end = end;
}
// Host-supplied segment when the game's own command memory runs out: when the game's out-of-memory callback cannot allocate from its command memory pool, it calls AddCommandMemory with an empty pool (p = NULL, n = 0)
// (opening the advanced stats page on the level 88 sorceress save: many list items in one frame, and the original implementation segfaulted here reading NULL+8). Our records are larger than real-device NVN commands, so the budget the game reserved for real-device usage is not necessarily enough,
// so this is not an error state of the game but something we must catch: provide a segment of host memory to keep recording. The segment is replayed in place at submit and reclaimed after being kept for 8 frames (submit is in the same frame as recording or the next one).
// (Do not use the whole game chunk as the range: SMBW sub-allocates many recordings out of one big chunk, and a later grant inside that chunk freed segments still in use.)
// Reclaim follows the game's command memory chunks, not a frame count: the game caches recorded command handles and CALLs them every frame until it re-records them itself (SMBW's object setup handles);
// once a handle was recorded into a host segment, the old "reclaim 8 frames after lending" rewrote the segment while the game still CALLed it (with overflow forced on the Mac, the replay read garbage commands: RGBA16F used as a depth format for a pipeline).
// A segment hangs off one recording through the JUMP at the end of its guest part: that recording occupies [BEGIN, end of JUMP) in guest memory, and the game only gives those bytes out again once they are dead
// (when it re-records that material in place; packing the next recording right behind by used never touches them, since used includes the JUMP). So a segment remembers that guest range and is reclaimed when the game gives out memory
// overlapping it; segments overflowing from a host segment keep the same range.
// The segment pool grows on demand (SMBW opening course: many small command buffers each overflow once per frame, and the old fixed 64 segments × 1 MB ran out within 8 frames); abort only beyond HOSTSEG_CAP in total.
// A segment is HOSTSEG_SZ, with two exceptions (hostseg_size): an empty pool with a length of at most HOSTSEG_SMALL outside a recording gets that much rounded up to HOSTSEG_SMALL
// (Super Mario Odyssey hands out many such 512-byte grants before recording; at 256 KB each, and with no recording to reclaim them by, they reached the cap entering the Wooded
// Kingdom), and a record larger than HOSTSEG_SZ gets a segment that holds it. A free segment is taken again only for a request of its own size; the cap counts every segment's bytes.
// An empty-pool grant outside a recording means the game's own allocator for command memory came back empty. On hardware a null pool cannot be written to, so a game only gets here
// when the host's records outgrew what it measured for them (Odyssey's "DisplayList" heap: docs/reference/memory.md, "Model heaps sized by the host's command record sizes").
// Those segments have no guest range to be reclaimed by, so they are not reclaimed; the cap's abort names the state, and the fix belongs in the record sizes
#define HOSTSEG_SZ (256u << 10)
#define HOSTSEG_SMALL (4u << 10)
#define HOSTSEG_CAP (1u << 30)
uint64_t dm_nvn_frame_count(void);
typedef struct { uint8_t *mem; size_t n; uint64_t frame; int busy; uintptr_t org_lo, org_hi; } hostseg_t;   // org: the guest range of the recording it hangs off (0 = unknown, never reclaimed)
static hostseg_t *g_hostseg; static int g_nhostseg, g_caphostseg; static uint64_t g_hostseg_bytes;
static pthread_mutex_t g_hostseg_mu = PTHREAD_MUTEX_INITIALIZER;
static size_t hostseg_size(const cmd_t *o, int empty_pool, size_t want) {
    if (empty_pool) return want && want <= HOSTSEG_SMALL && !o->recording ? HOSTSEG_SMALL : HOSTSEG_SZ;
    return want + 16 > HOSTSEG_SZ ? (want + 16 + 0xfff) & ~(size_t)0xfff : HOSTSEG_SZ;   // 16: room for the next JUMP (want holds OP_RESERVE already, this is slack for the alignment)
}
// Take a segment of n bytes: origin = the current recording's guest bytes [BEGIN, end of JUMP); if the write pointer is already in a host segment (overflowing again), keep that segment's origin
static uint8_t *hostseg_take(const cmd_t *o, size_t n) {
    static unsigned total; uint64_t f = dm_nvn_frame_count(); uint8_t *m = NULL;
    uintptr_t lo = 0, hi = 0, cur = o ? (uintptr_t)o->cur : 0;
    pthread_mutex_lock(&g_hostseg_mu);
    int in_seg = 0;   // by the segment's end too: a small segment can end where another one starts
    for (int i = 0; i < g_nhostseg && cur; i++) if (cur >= (uintptr_t)g_hostseg[i].mem && o->end == g_hostseg[i].mem + g_hostseg[i].n) { lo = g_hostseg[i].org_lo; hi = g_hostseg[i].org_hi; in_seg = 1; break; }
    if (!in_seg && o && o->recording && o->cur) { lo = o->recording == 1 && o->begin && (uintptr_t)o->begin <= cur ? (uintptr_t)o->begin : cur; hi = cur + 16; }
    for (int i = 0; i < g_nhostseg && !m; i++)
        if (!g_hostseg[i].busy && g_hostseg[i].n == n) { g_hostseg[i] = (hostseg_t){ g_hostseg[i].mem, n, f, 1, lo, hi }; m = g_hostseg[i].mem; }
    if (!m && g_hostseg_bytes + n <= HOSTSEG_CAP && (m = malloc(n))) {
        if (g_nhostseg == g_caphostseg) { g_caphostseg = g_caphostseg ? g_caphostseg * 2 : 64; g_hostseg = realloc(g_hostseg, sizeof *g_hostseg * (size_t)g_caphostseg); }
        g_hostseg[g_nhostseg++] = (hostseg_t){ m, n, f, 1, lo, hi }; g_hostseg_bytes += n;
    }
    unsigned t = ++total; int ns = g_nhostseg; uint64_t bytes = g_hostseg_bytes;
    pthread_mutex_unlock(&g_hostseg_mu);
    if (!m) die("NVN command memory: the game's pool is empty and host-lent segments reached %u MB without being reclaimed by the game's blocks", HOSTSEG_CAP >> 20);
    if (t == 1 || t % 1024 == 0) LOG("[nvn] the game's command memory ran short, the host lends a segment of %zu KB (%u times so far, frame %llu, segment pool %d segments, %llu KB)", n >> 10, t, (unsigned long long)f, ns, (unsigned long long)(bytes >> 10));
    return m;
}
// The game gives out [lo, hi): segments whose recording's guest bytes overlap it are dead, reclaim them
static void hostseg_release(uintptr_t lo, uintptr_t hi) {
    pthread_mutex_lock(&g_hostseg_mu);
    for (int i = 0; i < g_nhostseg; i++) if (g_hostseg[i].busy && g_hostseg[i].org_hi > lo && g_hostseg[i].org_lo < hi) g_hostseg[i].busy = 0;
    pthread_mutex_unlock(&g_hostseg_mu);
}
NVN(nvnCommandBufferAddCommandMemory, void, NVNcommandBuffer *c, const NVNmemoryPool *p, ptrdiff_t off, size_t n) {
    const pool_t *pl = (const pool_t *)p;
    if (!pl || n < 64) { if (cs_on()) { pthread_mutex_lock(&g_cs_mu); g_cs.hostseg++; pthread_mutex_unlock(&g_cs_mu); } size_t sz = hostseg_size(CB(c), !pl, n); cmd_new_segment(CB(c), hostseg_take(CB(c), sz), sz); return; }
    if (!pl->mem) { LOG("[RED] AddCommandMemory: pool has no CPU storage"); return; }
#ifndef NVN_CMDMEM_ADDR
#define NVN_CMDMEM_ADDR(a) ((uint8_t *)(a))
#endif
    hostseg_release((uintptr_t)NVN_CMDMEM_ADDR(pl->mem + off), (uintptr_t)NVN_CMDMEM_ADDR(pl->mem + off) + n);
    if (cs_on()) { pthread_mutex_lock(&g_cs_mu); g_cs.grants++; g_cs.grant_bytes += n; if (!g_cs.grant_min || n < g_cs.grant_min) g_cs.grant_min = n; if (n > g_cs.grant_max) g_cs.grant_max = n; pthread_mutex_unlock(&g_cs_mu); }
    { static int cblog = -1; if (cblog < 0) cblog = getenv("DM_NVN_CBLOG") != NULL; static uint64_t ng; if (cblog && (++ng <= 60 || ng % 500 == 0)) LOG("[nvn] AddCommandMemory #%llu cmd=%p pool %p +%#lx length %#zx (recording %d)", (unsigned long long)ng, (void *)c, (const void *)pl, (long)off, n, CB(c)->recording); }
    // NVN_CMDMEM_ADDR is the identity here; a trace replay harness defines it so records go to a shadow of the game's command memory, since it restores the pool bytes the real driver wrote
    // there; the shadow spreads the game's addresses (x4) because these records outgrow the NVN2 driver's commands (BindTexture: 24 bytes against 16-20) and the game
    // places each recording right behind the driver's last one
    nvn_direct_host_written((const uint8_t *)NVN_CMDMEM_ADDR(pl->mem + off), n);   // Lichii: the records are written in place: never write-tracked (nvn_tracking.c)
    cmd_new_segment(CB(c), NVN_CMDMEM_ADDR(pl->mem + off), n);
    CB(c)->used = 0;   // GetCommandMemoryUsed counts from the start of the memory the game last added, across recordings (see there)
}
NVN(nvnCommandBufferAddControlMemory, void, NVNcommandBuffer *c, void *m, size_t n) {
    (void)m;
    CB(c)->ctl_size = n;   // The host's recordings contain no control-memory entries; the entire last block remains free.
}
// Get space for one record and keep `keep` bytes after it (OP_RESERVE, 0 for the END); if not enough, call back into the game per the NVN convention to add command memory
static uint8_t *cmd_alloc_keep(cmd_t *o, uint32_t size, uint32_t keep) {
    size = (size + 7) & ~7u;
    for (int tries = 0; !o->cur || o->cur + size + keep > o->end; tries++) {
        // The game's callback cannot supply enough (newer-SDK games hand out small chunks sized for real-device commands, or disable the callback and manage memory themselves): continue recording in a host segment instead of aborting
        if (1) {   // Lichii: the game's callback is guest code, which a host call made from the JIT cannot run: a host-lent segment every time (upstream: !o->cb || tries == 2)
            static int once; if (!once++) LOG("[nvn] command memory: the game's callback did not add enough (cb=%p, need %u bytes), using a host-lent segment", (void *)o->cb, size + keep);
            size_t sz = hostseg_size(o, 0, size + keep); cmd_new_segment(o, hostseg_take(o, sz), sz); break;
        }
        if (cs_on()) { pthread_mutex_lock(&g_cs_mu); g_cs.cbs++; pthread_mutex_unlock(&g_cs_mu); }
        { static int cblog = -1; if (cblog < 0) cblog = getenv("DM_NVN_CBLOG") != NULL; static uint64_t ncb;
          if (cblog && (++ncb <= 40 || ncb % 1000 == 0)) LOG("[nvn] command memory callback #%llu cmd=%p need %u (used in this segment %ld)", (unsigned long long)ncb, (void *)o, size + keep, o->cur && o->end ? (long)(o->cur - (o->end - 1)) : -1L); }
        o->cb((NVNcommandBuffer *)o, NVN_COMMAND_BUFFER_MEMORY_EVENT_OUT_OF_COMMAND_MEMORY, size + keep, o->cb_data);
    }
    if (cs_on()) { pthread_mutex_lock(&g_cs_mu); g_cs.bytes += size; g_cs.recs++; pthread_mutex_unlock(&g_cs_mu); }
    uint8_t *p = o->cur; o->cur += size; o->used += size; put_hdr(p, 0, size);
    if (o->recording == 2) { o->begin = p; o->recording = 1; }   // The recording's first record is its handle
    return p;
}
static uint8_t *cmd_alloc(cmd_t *o, uint32_t size) { return cmd_alloc_keep(o, size, OP_RESERVE); }
// A recording starts with its first command, as in NVN: no BEGIN record, so an empty recording takes only its END. Games size command memory by GetCommandMemoryUsed
// (nvn_obj.h draw_rec): Breath of the Wild sums one scratch recording per mesh and per shape and render pass into each model's prerecorded commands, and with an
// 8-byte BEGIN on each of several hundred recordings a model came out a 4 KB page over its resource size table entry
NVN(nvnCommandBufferBeginRecording, void, NVNcommandBuffer *c) {
    cmd_t *o = CB(c);
    if (o->recording) LOG("[RED] BeginRecording re-entered");
    o->recording = 2;   // 2: recording, no command yet (cmd_alloc sets begin)
}
// An empty recording (nothing between Begin and End) takes no command memory and is a shared END, as the NVN2 driver's empty recordings take none: Mario Kart World
// records one per frame and puts the next recording, through the next AddCommandMemory, at the same address (by the driver's count). Its END at the start of that
// memory was overwritten and its handle then named the next list: a list that waits on an event ran twice, the first time ahead of the signals it waits for
// (queue event waits). The empty handle is only ever called or submitted, never written
static const uint64_t g_empty_list[2] = {OP_END | (uint64_t)8 << 32, 0};
NVN(nvnCommandBufferEndRecording, NVNcommandHandle, NVNcommandBuffer *c) {
    cmd_t *o = CB(c);
    if (o->recording == 2) { o->recording = 0; return (NVNcommandHandle)(uintptr_t)g_empty_list; }
    uint8_t *p = cmd_alloc_keep(o, 8, 0); put_hdr(p, OP_END, 8);   // nothing follows: the END may take the segment's last bytes
    o->recording = 0;
    return (NVNcommandHandle)o->begin;
}
NVN(nvnCommandBufferIsRecording, NVNboolean, const NVNcommandBuffer *c) { return ((const cmd_t *)c)->recording != 0; }
NVN(nvnCommandBufferCopyCommands, void, NVNcommandBuffer *c, int n, const NVNcommandHandle *h) {
    for (int i = 0; i < n; i++) { uint8_t *p = cmd_alloc(CB(c), 16); put_hdr(p, OP_CALL, 16); memcpy(p + 8, &h[i], 8); }
}
NVN(nvnCommandBufferSignalEvent, void, NVNcommandBuffer *c, const NVNevent *e, NVNeventSignalMode m, NVNeventSignalLocation loc, int flags, uint32_t v) {
    (void)loc; (void)flags;
    uint8_t *p = cmd_alloc(CB(c), 24); put_hdr(p, OP_SIGNAL_EVENT, 24);
    const ev_t *ev = (const ev_t *)e; volatile uint32_t *val = ev->val; uint32_t mm = m;
    memcpy(p + 8, &val, 8); memcpy(p + 16, &mm, 4); memcpy(p + 20, &v, 4);
}
// A fence recorded in a command buffer: the replay fences the sync with the submission that reaches this point (OP_FENCE_SYNC). Unreal Engine's NVN RHI fences its command buffers this way (Octopath Traveler 0); Pokémon Legends: Z-A fences once a frame this way and waits on it
NVN(nvnCommandBufferFenceSync, void, NVNcommandBuffer *c, NVNsync *s, NVNsyncCondition cond, int flags) {
    (void)cond; (void)flags;
    uint8_t *p = cmd_alloc(CB(c), 16); put_hdr(p, OP_FENCE_SYNC, 16); memcpy(p + 8, &s, 8);
}
NVN(nvnCommandBufferResetCounter, void, NVNcommandBuffer *c, NVNcounterType t) {
    uint8_t *p = cmd_alloc(CB(c), 16); put_hdr(p, OP_RESET_COUNTER, 16); uint64_t tt = t; memcpy(p + 8, &tt, 8);
}
NVN(nvnCommandBufferReportCounter, void, NVNcommandBuffer *c, NVNcounterType t, NVNbufferAddress a) {
    uint8_t *p = cmd_alloc(CB(c), 24); put_hdr(p, OP_REPORT_COUNTER, 24); uint64_t tt = t; memcpy(p + 8, &tt, 8); memcpy(p + 16, &a, 8);
}
NVN(nvnCommandBufferCopyBufferToBuffer, void, NVNcommandBuffer *c, NVNbufferAddress src, NVNbufferAddress dst, size_t n, int flags) {
    (void)flags;
    uint8_t *p = cmd_alloc(CB(c), 32); put_hdr(p, OP_COPY_B2B, 32); uint64_t nn = n;
    memcpy(p + 8, &src, 8); memcpy(p + 16, &dst, 8); memcpy(p + 24, &nn, 8);
}
// uniform update: the data is copied into command memory at record time (NVN semantics: the caller may modify the original data after recording) and written into the buffer at submit
NVN(nvnCommandBufferUpdateUniformBuffer, void, NVNcommandBuffer *c, NVNbufferAddress buf, size_t bufsz, ptrdiff_t off, size_t n, const void *data) {
    (void)bufsz;
    uint32_t sz = (uint32_t)(24 + n);
    uint8_t *p = cmd_alloc(CB(c), sz); put_hdr(p, OP_UPDATE_UB, (sz + 7) & ~7u);
    uint64_t dst = buf + off, nn = n; memcpy(p + 8, &dst, 8); memcpy(p + 16, &nn, 8); memcpy(p + 24, data, n);
}

// ---------- Render commands: parameters are recorded into command memory in the payload format of nvn_obj.h; for replay see nvn_mtl.m ----------
static void *rec(NVNcommandBuffer *c, uint32_t op, size_t payload) {
    uint32_t sz = (uint32_t)((8 + payload + 7) & ~(size_t)7);
    uint8_t *p = cmd_alloc(CB(c), sz); put_hdr(p, op, sz);
    memset(p + 8, 0, sz - 8);
    return p + 8;
}
#define REC(c, OP, T) ((T *)rec((c), (OP), sizeof(T)))
// Binding records with arrays are allocated by the actual count: prefix fields + n elements (the replay side reads only the first n). Previously they were recorded at fixed length for the maximum count (bind vertex buffers 272 bytes, bind textures / samplers 280 bytes,
// while there are usually only 1–2 in practice), and the empty tail took 40–50% of the command bytes per frame; the advanced stats page of the level 88 sorceress recorded 12.5 MB per frame, using up the game's own command memory pool (about 105 segments x 128 KB per frame)
#define RECN(c, OP, T, ARR, N) ((T *)rec((c), (OP), offsetof(T, ARR) + sizeof(((T *)0)->ARR[0]) * (size_t)(N)))
// Draws carry the indirect / multi-draw fields (ind onwards) only when they use them: a plain indexed draw record is 0x28 bytes instead of 0x40 (nvn_obj.h draw_rec)
static void rec_draw(NVNcommandBuffer *c, p_draw d) { size_t n = d.ind || d.mdi_cnt ? sizeof d : offsetof(p_draw, ind); memcpy(rec(c, OP_DRAW, n), &d, n); }
static void rec_draw_idx(NVNcommandBuffer *c, p_draw_idx d) { size_t n = d.ind || d.mdi_cnt ? sizeof d : offsetof(p_draw_idx, ind); memcpy(rec(c, OP_DRAW_IDX, n), &d, n); }
#if defined(__ANDROID__)
// NVN2 mesh draw: count task workgroups (or mesh workgroups when the program has no task shader) from first, with the bound program's task and mesh shaders. A draw like the
// others (it opens and joins render passes the same way); the Vulkan replay draws it with vkCmdDrawMeshTasksEXT (vka_draw.c). The signature is the game's calls' (switch2mac
// hle_nvn.c, MIT: x1 = 0, x2 = workgroups in Mario Kart World). The indirect, 3D and multi-draw variants are near the end of the file ("Mesh draws, texture clears ...")
NVN_NOPROTO(nvnCommandBufferDrawMeshTasks, void, NVNcommandBuffer *c, int first, int count) { rec_draw(c, (p_draw){NVN_PRIM_MESH, first, count, 1, 0, 0}); }
// Depth bounds test (Mario Kart World sets it ~1,300 times a frame): the Vulkan replay applies it as dynamic state (vka_draw.c); the Mac host keeps it empty (g_nops)
NVN(nvnCommandBufferSetDepthBounds, void, NVNcommandBuffer *c, uint32_t en, float lo, float hi) { *REC(c, OP_DEPTH_BOUNDS, p_depth_bounds) = (p_depth_bounds){(en & 0xff) != 0, lo, hi}; }
// A queue waits for an event another queue signals (Mario Kart World's two graphics queues hand five stages a frame back and forth this way, the waiting queue submitted
// first): the Vulkan replay holds the waiting queue's commands until the event's value is met in replay order and goes on with them there (vka_exec.c); the Mac host
// keeps it empty (g_nops)
NVN(nvnCommandBufferWaitEvent, void, NVNcommandBuffer *c, const NVNevent *e, NVNeventWaitMode m, uint32_t v) { *REC(c, OP_WAIT_EVENT, p_wait_event) = (p_wait_event){((const ev_t *)e)->val, (uint32_t)m, v}; }
#endif
static void copy_view(view_t *dst, uint8_t *has, const NVNtextureView *v) { *has = v != NULL; if (v) *dst = *(const view_t *)v; }
// Records with textures snapshot the texture objects (0x80 bytes each, after the record struct) and replay reads the snapshots: an NVN command no longer depends on the API object once recorded, and the game may finalize, free or reuse it before submitting.
// Unity (Engage) does so when it changes scenes; only the object pointer used to be recorded, and replay read other data (a texture 1869367126 high, a Metal assertion abort).
#define SNAP_OFF(T) ((sizeof(T) + 7) & ~(size_t)7)
#define REC_SNAP(c, OP, T, K) ((T *)rec((c), (OP), SNAP_OFF(T) + (size_t)(K) * sizeof(tex_t)))
static const tex_t *tex_snap(void *rec_, size_t off, int k, const NVNtexture *t, uint64_t ra) {
    if (!t) return NULL;
    tex_t *s = (tex_t *)((uint8_t *)rec_ + off) + k;
    memcpy(s, t, sizeof *s);
    // The object is invalid at record time (Unity finalizes and re-initializes the same render target object between recording and submitting): keep the object pointer as before and read it at submit, when it is usually valid again; replay still checks the magic
    if (s->magic != MAGIC('T', 'E', 'X', 0)) {
        static int red;
        if (red++ < 4) {   // the first 4 log the call site and the start of the object, to tell what the game passed
            module_t *m = module_of((void *)ra); const uint64_t *q = (const uint64_t *)t;
            LOG("[RED] texture object invalid while recording (uninitialized or already finalized): %p index %d, call site %s+%#llx, object start %llx %llx %llx %llx %llx %llx; read again at submit", (const void *)t, k,
                m ? m->name : "?", m ? (unsigned long long)(ra - (uint64_t)m->base) : (unsigned long long)ra,
                (unsigned long long)q[0], (unsigned long long)q[1], (unsigned long long)q[2], (unsigned long long)q[3], (unsigned long long)q[4], (unsigned long long)q[5]);
        }
        return (const tex_t *)t;
    }
    return s;
}
NVN(nvnCommandBufferSetRenderTargets, void, NVNcommandBuffer *c, int n, const NVNtexture *const *ct, const NVNtextureView *const *cv,
    const NVNtexture *dt, const NVNtextureView *dv) {
    if (n < 0 || n > 8) { LOG("[RED] SetRenderTargets count %d", n); n = n < 0 ? 0 : 8; }
    p_set_rt *r = REC_SNAP(c, OP_SET_RT, p_set_rt, n + 1); r->n = n;
    for (int i = 0; i < n; i++) { r->color[i] = ct ? tex_snap(r, SNAP_OFF(p_set_rt), i, ct[i], nvn_caller()) : NULL; copy_view(&r->cview[i], &r->has_cview[i], cv ? cv[i] : NULL); }
    r->depth = tex_snap(r, SNAP_OFF(p_set_rt), n, dt, nvn_caller()); copy_view(&r->dview, &r->has_dview, dv);
}
NVN(nvnCommandBufferClearColor, void, NVNcommandBuffer *c, int index, const float *rgba, int mask) {
    p_clear_color *r = REC(c, OP_CLEAR_COLOR, p_clear_color); r->index = index; r->mask = mask; memcpy(r->rgba, rgba, 16);
}
// Clears of integer targets (Paper Mario: The Thousand-Year Door). Lichii: recorded as the values' bits (ints), which the replay puts in the clear value as they are: as floats
// they were exact only up to 2^24 in magnitude, and 0xffffffff became 2^32, out of range for the conversion back on x86
NVN(nvnCommandBufferClearColori, void, NVNcommandBuffer *c, int index, const int *rgba, int mask) {
    p_clear_color *r = REC(c, OP_CLEAR_COLOR, p_clear_color); r->index = index; r->mask = mask; memcpy(r->rgba, rgba, 16); r->ints = 1;
}
NVN(nvnCommandBufferClearColorui, void, NVNcommandBuffer *c, int index, const uint32_t *rgba, int mask) {
    p_clear_color *r = REC(c, OP_CLEAR_COLOR, p_clear_color); r->index = index; r->mask = mask; memcpy(r->rgba, rgba, 16); r->ints = 2;
}
NVN(nvnCommandBufferClearDepthStencil, void, NVNcommandBuffer *c, float depth, uint32_t dmask, int stencil, int smask) {
    p_clear_ds *r = REC(c, OP_CLEAR_DS, p_clear_ds); *r = (p_clear_ds){depth, (int)B(dmask), stencil, smask};
}
NVN(nvnCommandBufferSetViewports, void, NVNcommandBuffer *c, int first, int n, const float *v) {
    if (n < 0 || first + n > 16) { LOG("[RED] SetViewports first=%d n=%d", first, n); return; }
    p_viewports *r = REC(c, OP_VIEWPORTS, p_viewports); r->first = first; r->n = n;
    for (int i = 0; i < n; i++) memcpy(r->v[i], v + 4 * i, 16);
}
NVN(nvnCommandBufferSetScissor, void, NVNcommandBuffer *c, int x, int y, int w, int h) { *REC(c, OP_SCISSOR, p_scissor) = (p_scissor){x, y, w, h}; }
NVN(nvnCommandBufferSetDepthRange, void, NVNcommandBuffer *c, float n, float f) { *REC(c, OP_DEPTH_RANGE, p_depth_range) = (p_depth_range){n, f}; }
NVN(nvnCommandBufferBindProgram, void, NVNcommandBuffer *c, const NVNprogram *p, int stages) {
    *REC(c, OP_BIND_PROGRAM, p_bind_program) = (p_bind_program){(const prog_t *)p, stages};
}
static void bind_state(NVNcommandBuffer *c, int kind, const void *st, size_t n) { p_bind_state *r = rec(c, OP_BIND_STATE, offsetof(p_bind_state, bytes) + n); r->kind = kind; memcpy(r->bytes, st, n); }   // only the state's own bytes (nvn_obj.h p_bind_state)
NVN(nvnCommandBufferBindBlendState, void, NVNcommandBuffer *c, const NVNblendState *st) { bind_state(c, 0, st, sizeof(blend_st)); }
NVN(nvnCommandBufferBindColorState, void, NVNcommandBuffer *c, const NVNcolorState *st) { bind_state(c, 1, st, sizeof(color_st)); }
NVN(nvnCommandBufferBindChannelMaskState, void, NVNcommandBuffer *c, const NVNchannelMaskState *st) { bind_state(c, 2, st, sizeof(chmask_st)); }
NVN(nvnCommandBufferBindMultisampleState, void, NVNcommandBuffer *c, const NVNmultisampleState *st) { bind_state(c, 3, st, sizeof(ms_st)); }
NVN(nvnCommandBufferBindPolygonState, void, NVNcommandBuffer *c, const NVNpolygonState *st) { bind_state(c, 4, st, sizeof(poly_st)); }
NVN(nvnCommandBufferBindDepthStencilState, void, NVNcommandBuffer *c, const NVNdepthStencilState *st) { bind_state(c, 5, st, sizeof(ds_st)); }
NVN(nvnCommandBufferBindVertexAttribState, void, NVNcommandBuffer *c, int n, const NVNvertexAttribState *st) {
    if (n < 0 || n > 16) { LOG("[RED] BindVertexAttribState n=%d", n); return; }
    p_bind_vattr *r = RECN(c, OP_BIND_VATTR, p_bind_vattr, st, n); r->first = 0; r->n = n; memcpy(r->st, st, 4 * n);
}
NVN(nvnCommandBufferBindVertexStreamState, void, NVNcommandBuffer *c, int n, const NVNvertexStreamState *st) {
    if (n < 0 || n > 16) { LOG("[RED] BindVertexStreamState n=%d", n); return; }
    p_bind_vstream *r = RECN(c, OP_BIND_VSTREAM, p_bind_vstream, st, n); r->first = 0; r->n = n; memcpy(r->st, st, 8 * n);
}
NVN(nvnCommandBufferBindVertexBuffers, void, NVNcommandBuffer *c, int first, int n, const NVNbufferRange *rg) {
    if (n < 0 || first + n > 16) { LOG("[RED] BindVertexBuffers first=%d n=%d", first, n); return; }
    if (n == 1 && (unsigned)first < 16 && rg[0].size <= UINT32_MAX) {
        *REC(c, OP_BIND_VBUF1, p_bind_vbuf1) = (p_bind_vbuf1){(uint32_t)first, (uint32_t)rg[0].size, rg[0].address}; return;
    }
    p_bind_vbuf *r = RECN(c, OP_BIND_VBUF, p_bind_vbuf, r, n); r->first = first; r->n = n; memcpy(r->r, rg, sizeof *rg * n);
}
NVN(nvnCommandBufferBindUniformBuffer, void, NVNcommandBuffer *c, NVNshaderStage st, int i, NVNbufferAddress a, size_t n) {
    if ((unsigned)st < 6 && (unsigned)i < 16 && n <= UINT32_MAX) {
        *REC(c, OP_BIND_UBO_SMALL, p_bind_ubo_small) = (p_bind_ubo_small){(uint16_t)st, (uint16_t)i, (uint32_t)n, a}; return;
    }
    *REC(c, OP_BIND_UBO, p_bind_buf) = (p_bind_buf){st, i, a, n};
}
NVN(nvnCommandBufferBindUniformBuffers, void, NVNcommandBuffer *c, NVNshaderStage st, int first, int n, const NVNbufferRange *rg) {
    for (int i = 0; i < n; i++) h_nvnCommandBufferBindUniformBuffer(c, st, first + i, rg[i].address, rg[i].size);
}
static void ssbo_slot_check(int st, int i) {   // the replay keeps 16 storage buffer slots a stage (NVN's limit); a higher one is not bound
    if (i >= 16) { static int n; if (n++ < 8) LOG("[RED] storage buffer bound at slot %d of stage %d: the host keeps 16 a stage, the bind is dropped", i, st); }
}
NVN(nvnCommandBufferBindStorageBuffer, void, NVNcommandBuffer *c, NVNshaderStage st, int i, NVNbufferAddress a, size_t n) {
    ssbo_slot_check(st, i);
    *REC(c, OP_BIND_SSBO, p_bind_buf) = (p_bind_buf){st, i, a, n};
}
NVN(nvnCommandBufferBindStorageBuffers, void, NVNcommandBuffer *c, NVNshaderStage st, int first, int n, const NVNbufferRange *rg) {
    for (int i = 0; i < n; i++) { ssbo_slot_check(st, first + i); *REC(c, OP_BIND_SSBO, p_bind_buf) = (p_bind_buf){st, first + i, rg[i].address, rg[i].size}; }
}
static p_bind_handles *bind_handles(NVNcommandBuffer *c, int op, int st, int first, int n, const uint64_t *h) {
    if (n < 0 || n > 32) { LOG("[RED] bind handle count %d", n); return NULL; }
    p_bind_handles *r = RECN(c, op, p_bind_handles, h, n); r->stage = st; r->first = first; r->n = n; r->defer = 0; r->src = 0;
    // The driver writes each handle into the c2 table as a 32-bit word (texture ID in bits 0-19, sampler ID in bits 20-31), so the GPU never sees the upper half of the 64-bit handle.
    // EA SPORTS FC 27 binds separate textures with bit 32 set (0x1000001d1): read whole, its sampler ID was 4096, beyond any pool, where the GPU reads sampler 0
    for (int i = 0; i < n; i++) r->h[i] = (uint32_t)h[i];
    {   // this command buffer may have bound a pool other than the first: move its IDs to that pool's part of the table (handle: texture ID in bits 0-19, sampler ID from bit 20)
        int tb = CB(c)->tex_base, sb = CB(c)->samp_base;
        for (int i = 0; i < n && (tb || sb); i++) {
            uint64_t v = r->h[i], tid = v & 0xfffff, sid = v >> 20;
            if (op == OP_BIND_SAMP) { if (sid) r->h[i] = (sid + (uint64_t)sb) << 20; }
            else { if (tid) tid += (uint64_t)tb; if (sid && op == OP_BIND_TEX) sid += (uint64_t)sb; r->h[i] = tid | sid << 20; }
        }
    }
    return r;
}
// Image (writable texture): handle = texture pool index (GetImageHandle returns it as-is; RegisterImage writes the same table as RegisterTexture)
NVN(nvnCommandBufferBindImages, void, NVNcommandBuffer *c, NVNshaderStage st, int first, int n, const NVNimageHandle *h) {
    bind_handles(c, OP_BIND_IMG, st, first, n, (const uint64_t *)h);
}
// Separate texture binds: on hardware they fill their own handle table (c2 0x168 + 8 slot), apart from BindTexture's, so they go to slots 128 + n (NVN_SEPTEX_SLOT0)
NVN(nvnCommandBufferBindSeparateTextures, void, NVNcommandBuffer *c, NVNshaderStage st, int first, int n, const NVNseparateTextureHandle *h) {
    bind_handles(c, OP_BIND_TEX, st, NVN_SEPTEX_SLOT0 + first, n, (const uint64_t *)h);
}
NVN(nvnCommandBufferBindSeparateTexture, void, NVNcommandBuffer *c, NVNshaderStage st, int i, NVNseparateTextureHandle h) { uint64_t v; memcpy(&v, &h, 8); bind_handles(c, OP_BIND_TEX, st, NVN_SEPTEX_SLOT0 + i, 1, &v); }   // single (Engage): as the plural version
NVN(nvnCommandBufferBindSeparateSamplers, void, NVNcommandBuffer *c, NVNshaderStage st, int first, int n, const NVNseparateSamplerHandle *h) {
    bind_handles(c, OP_BIND_SAMP, st, first, n, (const uint64_t *)h);
}
NVN(nvnCommandBufferBindSeparateSampler, void, NVNcommandBuffer *c, NVNshaderStage st, int i, NVNseparateSamplerHandle h) {
    bind_handles(c, OP_BIND_SAMP, st, i, 1, &h.value);
}
NVN(nvnCommandBufferSetStencilValueMask, void, NVNcommandBuffer *c, NVNface f, int m) { *REC(c, OP_STENCIL, p_stencil) = (p_stencil){f, m, 0, 0, 0}; }
NVN(nvnCommandBufferSetStencilMask, void, NVNcommandBuffer *c, NVNface f, int m) { *REC(c, OP_STENCIL, p_stencil) = (p_stencil){f, 0, m, 0, 1}; }
NVN(nvnCommandBufferSetStencilRef, void, NVNcommandBuffer *c, NVNface f, int v) { *REC(c, OP_STENCIL, p_stencil) = (p_stencil){f, 0, 0, v, 2}; }
NVN(nvnCommandBufferSetPolygonOffsetClamp, void, NVNcommandBuffer *c, float fa, float un, float cl) {
    *REC(c, OP_POLY_OFFSET, p_poly_offset) = (p_poly_offset){fa, un, cl};
}
NVN(nvnCommandBufferDrawArrays, void, NVNcommandBuffer *c, NVNdrawPrimitive pr, int first, int n) { rec_draw(c, (p_draw){pr, first, n, 1, 0, 0}); }
// Indirect: the {count, instances, first, base instance} record at ind is usually written by the GPU earlier in the frame (Hyrule Warriors: Age of Calamity's first battle), so the replay
// issues a Metal indirect draw from the pool buffer instead of reading it at record or replay time (nvn_mtl.m draw)
NVN(nvnCommandBufferDrawArraysIndirect, void, NVNcommandBuffer *c, NVNdrawPrimitive pr, NVNbufferAddress ind) { rec_draw(c, (p_draw){pr, 0, 0, 1, 0, ind}); }
NVN(nvnCommandBufferDrawElementsInstanced, void, NVNcommandBuffer *c, NVNdrawPrimitive pr, NVNindexType it, int n, NVNbufferAddress ib,
    int bv, int bi, int inst) {
    rec_draw_idx(c, (p_draw_idx){pr, it, n, bv, bi, inst, ib});
}
NVN(nvnCommandBufferDispatchCompute, void, NVNcommandBuffer *c, int x, int y, int z) {
    // The thunk is a tail jump (br), so the return address here is the call site in the game
    *REC(c, OP_DISPATCH, p_dispatch) = (p_dispatch){x, y, z, nvn_caller()};
}
// Indirect: the three group counts at ind are read by the GPU when the dispatch runs (nvn_mtl.m dispatch)
NVN(nvnCommandBufferDispatchComputeIndirect, void, NVNcommandBuffer *c, NVNbufferAddress ind) {
    *REC(c, OP_DISPATCH, p_dispatch) = (p_dispatch){0, 0, 0, nvn_caller(), ind};
}
NVN(nvnCommandBufferBarrier, void, NVNcommandBuffer *c, int bits) { *(int32_t *)rec(c, OP_BARRIER, 8) = bits; }
NVN(nvnCommandBufferSetTexturePool, void, NVNcommandBuffer *c, const NVNtexturePool *p) { REC(c, OP_SET_TEX_POOL, p_set_pool)->pool = (const dpool_t *)p; CB(c)->tex_base = dpool_base(g_tpb, g_ntpb, p); }
NVN(nvnCommandBufferSetSamplerPool, void, NVNcommandBuffer *c, const NVNsamplerPool *p) { REC(c, OP_SET_SAMP_POOL, p_set_pool)->pool = (const dpool_t *)p; CB(c)->samp_base = dpool_base(g_spb, g_nspb, p); }

NVN(nvnCommandBufferCopyBufferToTexture, void, NVNcommandBuffer *c, NVNbufferAddress src, const NVNtexture *t, const NVNtextureView *v,
    const NVNcopyRegion *r, int flags) {
    p_copy_b2t *o = REC_SNAP(c, OP_COPY_B2T, p_copy_b2t, 1); o->src = src; o->tex = tex_snap(o, SNAP_OFF(p_copy_b2t), 0, t, nvn_caller()); copy_view(&o->view, &o->has_view, v); o->r = *r; o->flags = flags;
}
NVN(nvnCommandBufferCopyTextureToBuffer, void, NVNcommandBuffer *c, const NVNtexture *t, const NVNtextureView *v, const NVNcopyRegion *r,
    NVNbufferAddress dst, int flags) {
    p_copy_b2t *o = REC_SNAP(c, OP_COPY_T2B, p_copy_b2t, 1); o->src = dst; o->tex = tex_snap(o, SNAP_OFF(p_copy_b2t), 0, t, nvn_caller()); copy_view(&o->view, &o->has_view, v); o->r = *r; o->flags = flags;
}
// Clear a texture region with the given value (float / unsigned integer versions; mask = channel write mask)
NVN(nvnCommandBufferClearTexture, void, NVNcommandBuffer *c, const NVNtexture *t, const NVNtextureView *v, const NVNcopyRegion *r, const float *rgba, int mask) {
    p_clear_tex *o = REC_SNAP(c, OP_CLEAR_TEX, p_clear_tex, 1); o->tex = tex_snap(o, SNAP_OFF(p_clear_tex), 0, t, nvn_caller()); copy_view(&o->view, &o->has_view, v); o->r = *r; o->mask = mask; o->is_uint = 0; memcpy(o->c.f, rgba, 16);
}
NVN(nvnCommandBufferClearTextureui, void, NVNcommandBuffer *c, const NVNtexture *t, const NVNtextureView *v, const NVNcopyRegion *r, const uint32_t *rgba, int mask) {
    p_clear_tex *o = REC_SNAP(c, OP_CLEAR_TEX, p_clear_tex, 1); o->tex = tex_snap(o, SNAP_OFF(p_clear_tex), 0, t, nvn_caller()); copy_view(&o->view, &o->has_view, v); o->r = *r; o->mask = mask; o->is_uint = 1; memcpy(o->c.u, rgba, 16);
}
NVN(nvnCommandBufferCopyTextureToTexture, void, NVNcommandBuffer *c, const NVNtexture *s, const NVNtextureView *sv, const NVNcopyRegion *sr,
    const NVNtexture *d, const NVNtextureView *dv, const NVNcopyRegion *dr, int flags) {
    p_copy_t2t *o = REC_SNAP(c, OP_COPY_T2T, p_copy_t2t, 2); o->src = tex_snap(o, SNAP_OFF(p_copy_t2t), 0, s, nvn_caller()); o->dst = tex_snap(o, SNAP_OFF(p_copy_t2t), 1, d, nvn_caller());
    copy_view(&o->sv, &o->has_sv, sv); copy_view(&o->dv, &o->has_dv, dv); o->sr = *sr; o->dr = *dr; o->flags = flags;
}
// A textured rectangle into the bound colour targets, replayed as draw_tex (nvn_mtl.m). The handle's IDs are moved to the pools this command buffer bound, as bind_handles does:
// Streets of Rage 4 draws its movie frames into its scene target this way, from a second texture pool. Only the handle's lower 32 bits reach the GPU (bind_handles)
NVN(nvnCommandBufferDrawTexture, void, NVNcommandBuffer *c, NVNtextureHandle h, const NVNdrawTextureRegion *d, const NVNdrawTextureRegion *s) {
    uint64_t tid = (uint32_t)h & 0xfffff, sid = (uint32_t)h >> 20;
    if (tid) tid += (uint64_t)CB(c)->tex_base;
    if (sid) sid += (uint64_t)CB(c)->samp_base;
    *REC(c, OP_DRAW_TEX, p_draw_tex) = (p_draw_tex){tid | sid << 20, *d, *s};
}
NVN(nvnCommandBufferSetCopyRowStride, void, NVNcommandBuffer *c, ptrdiff_t st) { p_copy_stride *o = REC(c, OP_COPY_STRIDE, p_copy_stride); o->stride = st; o->image = 0; CB(c)->copy_row = st; }
NVN(nvnCommandBufferSetCopyImageStride, void, NVNcommandBuffer *c, ptrdiff_t st) { p_copy_stride *o = REC(c, OP_COPY_STRIDE, p_copy_stride); o->stride = st; o->image = 1; CB(c)->copy_image = st; }   // Link's Awakening
NVN(nvnCommandBufferSetBlendColor, void, NVNcommandBuffer *c, const float *v) { memcpy(REC(c, OP_BLEND_COLOR, p_blend_color)->rgba, v, 16); }

// ---------- State objects: packed in place (nvn_obj.h) ----------
#define ST(T, x) ((T *)(x))
NVN(nvnBlendStateSetDefaults, void, NVNblendState *s) { *ST(blend_st, s) = (blend_st){0, NVN_BLEND_FUNC_ONE, NVN_BLEND_FUNC_ZERO, NVN_BLEND_FUNC_ONE, NVN_BLEND_FUNC_ZERO, NVN_BLEND_EQUATION_ADD, NVN_BLEND_EQUATION_ADD}; }
NVN(nvnBlendStateSetBlendTarget, void, NVNblendState *s, int t) { ST(blend_st, s)->target = t; }
NVN(nvnBlendStateSetBlendFunc, void, NVNblendState *s, NVNblendFunc sr, NVNblendFunc dr, NVNblendFunc sa, NVNblendFunc da) {
    blend_st *o = ST(blend_st, s); o->src_rgb = sr; o->dst_rgb = dr; o->src_a = sa; o->dst_a = da;
}
NVN(nvnBlendStateSetBlendEquation, void, NVNblendState *s, NVNblendEquation er, NVNblendEquation ea) { ST(blend_st, s)->eq_rgb = er; ST(blend_st, s)->eq_a = ea; }
// Advanced blend modes (NV_blend_equation_advanced): kept in the state, and while one is set the hardware ignores the blend functions. The replay blends the Porter-Duff modes with
// the fixed-function factors they amount to (adv_blend in nvn_mtl.m); multiply, screen, overlay and the rest have no Metal blend state. Mode 0 (none, which Luigi's Mansion 3 sets
// on every blend state) is the default; Mario vs. Donkey Kong 1.0.1 sets SRC_OVER (8) on one state in its levels
NVN(nvnBlendStateSetAdvancedMode, void, NVNblendState *s, NVNblendAdvancedMode mode) { ST(blend_st, s)->adv = (unsigned)mode <= NVN_BLEND_ADVANCED_MODE_HSL_LUMINOSITY ? mode : 0; }
NVN(nvnBlendStateSetAdvancedPremultipliedSrc, void, NVNblendState *s, uint32_t pm) { ST(blend_st, s)->adv_npm = !B(pm); }
NVN(nvnBlendStateSetAdvancedOverlap, void, NVNblendState *s, NVNblendAdvancedOverlap o) { ST(blend_st, s)->adv_ovl = (unsigned)o & 3; }
NVN(nvnColorStateSetDefaults, void, NVNcolorState *s) { *ST(color_st, s) = (color_st){0, NVN_LOGIC_OP_COPY, NVN_ALPHA_FUNC_ALWAYS}; }
NVN(nvnColorStateSetBlendEnable, void, NVNcolorState *s, int i, uint32_t en) {
    color_st *o = ST(color_st, s); if (i < 0 || i > 7) return;
    o->blend_en = B(en) ? (o->blend_en | 1u << i) : (o->blend_en & ~(1u << i));
}
NVN(nvnChannelMaskStateSetDefaults, void, NVNchannelMaskState *s) { ST(chmask_st, s)->mask = 0xffffffffu; }
NVN(nvnChannelMaskStateSetChannelMask, void, NVNchannelMaskState *s, int i, uint32_t r, uint32_t g, uint32_t b, uint32_t a) {
    if (i < 0 || i > 7) return;
    uint32_t m = (B(r) ? 1 : 0) | (B(g) ? 2 : 0) | (B(b) ? 4 : 0) | (B(a) ? 8 : 0);
    chmask_st *o = ST(chmask_st, s); o->mask = (o->mask & ~(0xfu << (4 * i))) | m << (4 * i);
}
NVN(nvnMultisampleStateSetDefaults, void, NVNmultisampleState *s) { memset(s, 0, sizeof(ms_st)); ST(ms_st, s)->enable = 1; memset(ST(ms_loc_st, s)->loc, 0x88, sizeof ST(ms_loc_st, s)->loc); }
NVN(nvnMultisampleStateSetMultisampleEnable, void, NVNmultisampleState *s, uint32_t e) { ST(ms_st, s)->enable = B(e) != 0; }
NVN(nvnMultisampleStateSetSamples, void, NVNmultisampleState *s, int n) { ST(ms_st, s)->samples = (uint8_t)n; }
NVN(nvnMultisampleStateSetAlphaToCoverageEnable, void, NVNmultisampleState *s, uint32_t e) { ST(ms_st, s)->a2c = B(e) != 0; }
NVN(nvnMultisampleStateSetSampleLocationsEnable, void, NVNmultisampleState *s, uint32_t e) { ST(ms_st, s)->sl_en = B(e) != 0; }
// Kept as the driver keeps them (ms_loc_st): each coordinate rounded to 1/16 pixel and clamped to 0..15/16. The replay does not program sample locations
static uint8_t ms_loc_nibble(float v) { int n = (int)((double)(v * 16.f) + 0.5); return (uint8_t)(n < 0 ? 0 : n > 15 ? 15 : n); }
NVN(nvnMultisampleStateSetSampleLocations, void, NVNmultisampleState *s, int start, int n, const float *v) {
    for (int i = 0; i < n; i++) if ((unsigned)(start + i) < 16) ST(ms_loc_st, s)->loc[start + i] = (uint8_t)(ms_loc_nibble(v[2 * i]) | ms_loc_nibble(v[2 * i + 1]) << 4);
}
NVN(nvnMultisampleStateGetSamples, int, const NVNmultisampleState *s) { return ((const ms_st *)s)->samples; }
NVN(nvnMultisampleStateGetMultisampleEnable, NVNboolean, const NVNmultisampleState *s) { return ((const ms_st *)s)->enable; }
NVN(nvnPolygonStateSetDefaults, void, NVNpolygonState *s) { *ST(poly_st, s) = (poly_st){NVN_FACE_NONE, NVN_FRONT_FACE_CCW, NVN_POLYGON_MODE_FILL, 0}; }
NVN(nvnPolygonStateSetCullFace, void, NVNpolygonState *s, NVNface f) { ST(poly_st, s)->cull = f; }
NVN(nvnPolygonStateSetFrontFace, void, NVNpolygonState *s, NVNfrontFace f) { ST(poly_st, s)->front = f; }
NVN(nvnPolygonStateSetPolygonMode, void, NVNpolygonState *s, NVNpolygonMode m) { ST(poly_st, s)->mode = m; }
NVN(nvnPolygonStateSetPolygonOffsetEnables, void, NVNpolygonState *s, int e) { ST(poly_st, s)->offset_en = e; }
NVN(nvnDepthStencilStateSetDefaults, void, NVNdepthStencilState *s) {
    *ST(ds_st, s) = (ds_st){0, 1, NVN_DEPTH_FUNC_LESS, 0, NVN_STENCIL_FUNC_ALWAYS, NVN_STENCIL_FUNC_ALWAYS,
                            NVN_STENCIL_OP_KEEP, NVN_STENCIL_OP_KEEP, NVN_STENCIL_OP_KEEP, NVN_STENCIL_OP_KEEP, NVN_STENCIL_OP_KEEP, NVN_STENCIL_OP_KEEP};
}
NVN(nvnDepthStencilStateSetDepthTestEnable, void, NVNdepthStencilState *s, uint32_t e) { ST(ds_st, s)->dt = B(e) != 0; }
NVN(nvnDepthStencilStateSetDepthWriteEnable, void, NVNdepthStencilState *s, uint32_t e) { ST(ds_st, s)->dw = B(e) != 0; }
NVN(nvnDepthStencilStateSetDepthFunc, void, NVNdepthStencilState *s, NVNdepthFunc f) { ST(ds_st, s)->df = f; }
NVN(nvnDepthStencilStateSetStencilTestEnable, void, NVNdepthStencilState *s, uint32_t e) { ST(ds_st, s)->st = B(e) != 0; }
NVN(nvnDepthStencilStateSetStencilFunc, void, NVNdepthStencilState *s, NVNface f, NVNstencilFunc fn) {
    ds_st *o = ST(ds_st, s); if (f & NVN_FACE_FRONT) o->sf_f = fn; if (f & NVN_FACE_BACK) o->sf_b = fn;
}
NVN(nvnDepthStencilStateSetStencilOp, void, NVNdepthStencilState *s, NVNface f, NVNstencilOp fail, NVNstencilOp zfail, NVNstencilOp zpass) {
    ds_st *o = ST(ds_st, s);
    if (f & NVN_FACE_FRONT) { o->fail_f = fail; o->zfail_f = zfail; o->zpass_f = zpass; }
    if (f & NVN_FACE_BACK) { o->fail_b = fail; o->zfail_b = zfail; o->zpass_b = zpass; }
}
NVN(nvnVertexAttribStateSetDefaults, void, NVNvertexAttribState *s) { *ST(va_st, s) = (va_st){0, 0, 0}; }
NVN(nvnVertexAttribStateSetFormat, void, NVNvertexAttribState *s, NVNformat f, ptrdiff_t off) {
    if (off < 0 || off >= (1 << 19)) LOG("[RED] vertex attribute offset %td exceeds the packed range", off);
    ST(va_st, s)->fmt = f; ST(va_st, s)->offset = (uint32_t)off;
}
NVN(nvnVertexAttribStateSetStreamIndex, void, NVNvertexAttribState *s, int i) { ST(va_st, s)->stream = i; }
NVN(nvnVertexStreamStateSetDefaults, void, NVNvertexStreamState *s) { *ST(vs_st, s) = (vs_st){0, 0}; }
NVN(nvnVertexStreamStateSetStride, void, NVNvertexStreamState *s, ptrdiff_t st) { ST(vs_st, s)->stride = (uint32_t)st; }
NVN(nvnVertexStreamStateSetDivisor, void, NVNvertexStreamState *s, int d) { ST(vs_st, s)->divisor = d; }
// Getters: Monster Hunter Rise reads its stream states back when it builds a draw (about 24,000 calls in its character creation)
NVN(nvnVertexStreamStateGetStride, ptrdiff_t, const NVNvertexStreamState *s) { return (ptrdiff_t)((const vs_st *)s)->stride; }
NVN(nvnVertexStreamStateGetDivisor, int, const NVNvertexStreamState *s) { return ((const vs_st *)s)->divisor; }
NVN(nvnVertexAttribStateGetFormat, void, const NVNvertexAttribState *s, NVNformat *f, ptrdiff_t *off) { const va_st *o = (const va_st *)s; if (f) *f = (NVNformat)o->fmt; if (off) *off = (ptrdiff_t)o->offset; }
NVN(nvnVertexAttribStateGetStreamIndex, int, const NVNvertexAttribState *s) { return ((const va_st *)s)->stream; }

static _Atomic uint64_t g_sub_wall_ns, g_sub_cpu_ns, g_sub_instr;   // Wall clock / this thread's CPU time / this thread's retired instruction count for submit (excluding present) (taken away per frame)
static _Atomic uint64_t g_cum_sub_instr, g_cum_sub_cpu_ns;   // Cumulative versions of the above (the "translation" summary row every 300 frames takes the difference; on a real device read through the log port: instructions per draw / submit CPU per frame)
#if PLAT_APPLE
extern int thread_selfcounts(int type, void *buf, size_t nbytes);   // This thread's cumulative {instructions, cycles} (libSystem private interface, present on both Mac and iOS)
#else
static int thread_selfcounts(int type, void *buf, size_t nbytes) { (void)type; (void)buf; (void)nbytes; return -1; }   // no counters on Android yet (perf_event_open): they read 0
#endif
static uint64_t thr_instr(void) { uint64_t c[2] = {0, 0}; thread_selfcounts(1, c, sizeof c); return c[0]; }
// Supplementary readings for the submit path (on a real device Metal 4 instruction count is -18% while submit CPU time is +49%; need to tell apart kernel mode, execution efficiency, or frequency throttling):
// this thread's cycle count (the second item of the same interface), kernel-mode CPU time (THREAD_BASIC_INFO), wall clock
static void thr_counts(uint64_t *instr, uint64_t *cyc) { uint64_t c[2] = {0, 0}; thread_selfcounts(1, c, sizeof c); *instr = c[0]; *cyc = c[1]; }
static uint64_t thr_sys_ns(void) { return (uint64_t)plat_thread_self_system_ns(); }
static _Atomic uint64_t g_cum_sub_cyc, g_cum_sub_sys_ns, g_cum_sub_wall_ns;
static int frame_rec(void);
static _Atomic uint64_t g_nvn_ns;   // Host time spent in submit / present this frame (replay encoding + present), used by the timing instrumentation

// ---------- Lifecycle: dm_gfx_suspend (called before iOS goes to the background; equivalent to Switch suspend) ----------
// on=1: subsequent submits / presents block until on=0; before returning, wait for in-progress submit / present calls to exit and for all in-flight GPU work to complete (iOS requires no GPU work in the background).
static pthread_mutex_t g_susp_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_susp_cv = PTHREAD_COND_INITIALIZER;
static int g_susp, g_inflight_api;
__thread _Atomic int dm_host_work;   // see dm.h (defined here: the graphics unit tests compile this file but not hle_thread.c)
static void gfx_enter(void) { pthread_mutex_lock(&g_susp_mu); while (g_susp) pthread_cond_wait(&g_susp_cv, &g_susp_mu); g_inflight_api++; pthread_mutex_unlock(&g_susp_mu); dm_host_work++; }
static void gfx_leave(void) { dm_host_work--; pthread_mutex_lock(&g_susp_mu); g_inflight_api--; pthread_cond_broadcast(&g_susp_cv); pthread_mutex_unlock(&g_susp_mu); }
void dm_gfx_enter(void) { gfx_enter(); }   // the same gate for the Vulkan layer's submits and presents (dm/gfx/vk)
void dm_gfx_leave(void) { gfx_leave(); }
__attribute__((weak)) void dm_keys_stop(void) {}   // hle_hid.c: drop the remaining DM_KEYS steps (DM_MARK_KEYS_STOP; the graphics unit tests have no scripted input)
__attribute__((weak)) void dm_vk_drain(void) {}   // dm/gfx/vk/hle_vk.m: wait for the Vulkan layer's queue (the graphics unit tests do not link it)
__attribute__((weak)) void dm_gl_drain(void) {}   // dm/gfx/gl/hle_gl.m: the same for the GL layer
void dm_gfx_suspend(int on) {
    pthread_mutex_lock(&g_susp_mu);
    g_susp = on != 0;
    if (on) while (g_inflight_api) pthread_cond_wait(&g_susp_cv, &g_susp_mu);
    else pthread_cond_broadcast(&g_susp_cv);
    pthread_mutex_unlock(&g_susp_mu);
    if (on) { dm_vk_drain(); dm_gl_drain(); mtl_drain(); LOG("[nvn] graphics suspended: in-flight GPU work drained"); } else LOG("[nvn] graphics resumed");   // the Vulkan and GL queues first: presents on the NVN queue wait for their frames
}
// One submission at a time: the Metal replay keeps one command buffer and one upload batch for the submission in progress (nvn_mtl.m g_cb). Fire Emblem: Three Houses initializes two queues and
// submits on both from two threads; the second submission arrived while the first was replaying and the host stopped with "submit re-entered" at the ninth frame. NVN orders work on different queues only
// through syncs, so replaying the two queues' submissions in lock order is one valid order.
// The first submission's prepare step (mtl_prepare: the prewarm and the pipeline prebuild) loads shaders and builds pipelines and submits no GPU work, so it runs under the lock but outside the
// graphics gate: inside it, going to the background meanwhile (dm_gfx_suspend, on the main thread) waited for it, minutes on a first start. It stays host work for the guest scheduler. The lock
// comes before the gate: a second submission waiting for it must not count as in flight, or the suspend waits for it while the first waits to enter the closed gate holding the lock
NVN(nvnQueueSubmitCommands, void, NVNqueue *q, int n, const NVNcommandHandle *h) {
    dm_host_mutex_lock(&g_sub_mu);
    dm_host_work++; mtl_prepare(); dm_host_work--;
    gfx_enter();
    uint64_t t0 = now_ns(), c0 = plat_clock_ns(CLOCK_THREAD_CPUTIME_ID), i0, y0, s0 = thr_sys_ns(); thr_counts(&i0, &y0);
    ((queue_t *)q)->submits++;
    mtl_queue_activate(((const queue_t *)q)->serial);
    mtl_set_queue_origin_lower_left(((const queue_t *)q)->origin_ll);
    nvn_cap_submit(n, (const uint64_t *)h);   // DM_NVN_CAPTURE (nvn_capture.inc)
    mtl_sub_begin();
    for (int i = 0; i < ((queue_t *)q)->nwait; i++) mtl_sub_wait(((queue_t *)q)->wait_fence[i]);
    ((queue_t *)q)->nwait = 0;
    nvn_ft_poll();
    mtl_sub_index((const uint64_t *)h, n);
    for (int i = 0; i < n; i++) { mtl_exec_rest((const uint64_t *)h + i + 1, n - i - 1); mtl_exec((const uint8_t *)h[i]); }
    mtl_exec_rest(NULL, 0);
    ((queue_t *)q)->last_fence = mtl_sub_commit();
    if (sync_log()) LOG("[nvn] sync log: Submit queue %p (%d command buffers) -> fence %llu", (void *)q, n, (unsigned long long)((queue_t *)q)->last_fence);
    pthread_mutex_unlock(&g_sub_mu);
    uint64_t dw = now_ns() - t0;
    atomic_fetch_add(&g_nvn_ns, dw); atomic_fetch_add(&g_sub_wall_ns, dw);
    uint64_t i1, y1; thr_counts(&i1, &y1); uint64_t dc = plat_clock_ns(CLOCK_THREAD_CPUTIME_ID) - c0, di = i1 - i0;
    atomic_fetch_add(&g_cum_sub_cyc, y1 - y0); atomic_fetch_add(&g_cum_sub_sys_ns, thr_sys_ns() - s0); atomic_fetch_add(&g_cum_sub_wall_ns, dw);
    atomic_fetch_add(&g_sub_cpu_ns, dc); atomic_fetch_add(&g_cum_sub_cpu_ns, dc);
    atomic_fetch_add(&g_sub_instr, di); atomic_fetch_add(&g_cum_sub_instr, di);
    gfx_leave();
}

// Commands that only affect GPU state / only execute on the GPU: the null device only counts them (the thunk has already counted)
static void nvn_nop(void) {}
void *nvn_nop_fn(void) { return (void *)nvn_nop; }   // Lichii: the dispatch calls the empty implementations without a prototype directly
static const char *const g_nops[] = {
#if !defined(__ANDROID__)
    "nvnCommandBufferSetDepthBounds",   // (recorded on Android: OP_DEPTH_BOUNDS)
    "nvnCommandBufferWaitEvent",   // (recorded on Android: OP_WAIT_EVENT)
#endif
    "nvnWindowFinalize",
};
void nvn_nops_register(void) {   // Lichii: called by nvn_direct_init (upstream: a constructor)
    // Empty implementations overwrite the unimplemented stubs after the slot table is initialized; a misspelled name (not in main's name table) reports a failure and aborts at startup
    nvn_init_slots();
    for (unsigned k = 0; k < sizeof g_nops / sizeof *g_nops; k++) {
        unsigned i = 0;
        while (i < NVN_NFUNC && strcmp(nvn_names[i], g_nops[k])) i++;
        if (i == NVN_NFUNC) die("NVN empty implementation %s is not in main's name table", g_nops[k]);
        if (nvn_slot[i].fn != (void *)nvn_unimpl_entry) die("NVN %s has an implementation and is also listed as an empty implementation", g_nops[k]);
        nvn_slot[i].fn = (void *)nvn_nop;
    }
    unsigned nnop = sizeof g_nops / sizeof *g_nops;
    // Implemented functions other than command recording (nvnCommandBuffer*, tens of thousands per frame, where the host only appends records) run on the big host stack: submit / present / object creation do Metal work, and a game thread's stack may be too small (see nvn_entry.S nvn_big_entry)
    unsigned nbig = 0;
    for (unsigned i = 0; i < NVN_NFUNC; i++)
        if (nvn_slot[i].fn != (void *)nvn_unimpl_entry && nvn_slot[i].fn != (void *)nvn_nop && strncmp(nvn_names[i], "nvnCommandBuffer", 16)) {
            nvn_big_real[i] = nvn_slot[i].fn; nvn_slot[i].fn = (void *)nvn_big_entry; nbig++;
        }
    // DM_NVN_PROF=1: implemented functions (except unimplemented stubs and present: present includes the display-link wait and is counted separately) get a timing wrapper, used to break down the time spent in the "translation layer recording phase"
    if (getenv("DM_NVN_PROF"))
        for (unsigned i = 0; i < NVN_NFUNC; i++)
            if (nvn_slot[i].fn != (void *)nvn_unimpl_entry && strcmp(nvn_names[i], "nvnQueuePresentTexture")) { nvn_real[i] = nvn_slot[i].fn; nvn_slot[i].fn = (void *)nvn_prof_entry; }
    LOG("[nvn] null device: %u names, %u implemented (%u semantic + %u empty, %u of them on the large host stack), the other %u go to the red-reporting stub",
        NVN_NFUNC, g_nsem + nnop, g_nsem, nnop, nbig, NVN_NFUNC - g_nsem - nnop);
}

// ---------- Window / present ----------
// Lichii: upstream's window, present pacing and frame statistics (from here to the state-only interfaces) are left out: they sleep on the calling thread, assume one
// acquire thread and present through vka_present.c. Lichii's own window and present are src/video_core/nvn_direct/nvn_window.c

// ---------- State-only interfaces used by newer-SDK games (SMBW): buffer objects, pool / sampler / texture queries, views, color state, device and window ----------
// They only read and write the object itself and record no commands. Where the host has no matching capability (texture channel swizzle, depth / stencil sampling mode, sampler reduction, virtual mapping) the first occurrence logs one line and the call returns normally.
#define ONCE_LOG(...) do { static int once_; if (!once_++) LOG("[nvn] " __VA_ARGS__); } while (0)

// Buffer object = pool + offset + size; the GPU address and the CPU mapping are both the pool's host pointer + offset (same convention as MemoryPoolGetBufferAddress)
typedef struct { NVNdevice *dev; const pool_t *pool; int64_t off; uint64_t size; } buf_builder_t;   FITS(buf_builder_t, NVNbufferBuilder);
// Guest ring allocators write offset/size directly at +0x18/+0x20 (issue #723).
// The intervening driver bytes have no host meaning; keep them reserved.
typedef struct { uint32_t magic; const pool_t *pool; uint8_t reserved[8]; int64_t off; uint64_t size; } buf_t; FITS(buf_t, NVNbuffer);
_Static_assert(offsetof(buf_t, off) == 0x18, "NVNbuffer guest offset");
_Static_assert(offsetof(buf_t, size) == 0x20, "NVNbuffer guest size");
#define BB(b) ((buf_builder_t *)(b))
#define BF(b) ((const buf_t *)(b))
NVN(nvnBufferBuilderSetDefaults, void, NVNbufferBuilder *b) { NVNdevice *d = BB(b)->dev; memset(b, 0, sizeof(buf_builder_t)); BB(b)->dev = d; }
NVN(nvnBufferBuilderSetDevice, void, NVNbufferBuilder *b, NVNdevice *d) { BB(b)->dev = d; }
NVN(nvnBufferBuilderSetStorage, void, NVNbufferBuilder *b, NVNmemoryPool *p, ptrdiff_t off, size_t n) { BB(b)->pool = (const pool_t *)p; BB(b)->off = off; BB(b)->size = n; }
NVN(nvnBufferInitialize, NVNboolean, NVNbuffer *buf, const NVNbufferBuilder *b) {
    const buf_builder_t *bb = (const buf_builder_t *)b;
    if (!bb->pool || !bb->pool->mem) { LOG("[RED] BufferInitialize: pool has no storage (pool %p flags %#x offset %#llx size %#llx)", (const void *)bb->pool, bb->pool ? bb->pool->flags : 0, (unsigned long long)bb->off, (unsigned long long)bb->size); return 0; }
    // The NVN driver does not check a buffer's range against its pool, and the host's own GPU access never goes by the buffer: it takes an address and a length from the command (mtl_addr resolves it
    // in the pool's memory and reports a range that leaves the pool as its own red line). Breath of the Wild creates a 256 KB buffer over a 144 KB pool for a screen readback that feeds a
    // compressing encoder (when the pause menu opens, and near a save commit): the buffer's one use is a texture-to-buffer copy of the picture's own extent, which lies inside the pool. Not an error by itself: one line.
    if ((uint64_t)bb->off + bb->size > bb->pool->size) ONCE_LOG("BufferInitialize: a buffer reaches past the end of its pool (offset %#llx size %#llx pool %#llx); only an access that leaves the pool is an error, and it is reported where it happens", (unsigned long long)bb->off, (unsigned long long)bb->size, (unsigned long long)bb->pool->size);
    *(buf_t *)buf = (buf_t){.magic = MAGIC('B', 'U', 'F', 0), .pool = bb->pool, .off = bb->off, .size = bb->size};
    pc_use(bb->pool, 'b', bb->off, bb->size); mtl_pool_need(bb->pool->mem + bb->off, bb->size, "a buffer");   // the GPU reaches a buffer by its address
    OLOG("BufferInitialize %p pool=%p off=%#llx size=%#llx", (void *)buf, (const void *)bb->pool, (unsigned long long)bb->off, (unsigned long long)bb->size);
    return 1;
}
NVN(nvnBufferFinalize, void, NVNbuffer *buf) { ((buf_t *)buf)->magic = 0; }
NVN(nvnBufferSetDebugLabel, void, NVNbuffer *buf, const char *s) { (void)buf; (void)s; }
NVN(nvnBufferGetAddress, NVNbufferAddress, const NVNbuffer *buf) { mtl_pool_need(BF(buf)->pool->mem + BF(buf)->off, BF(buf)->size, "a buffer's address"); return (NVNbufferAddress)(BF(buf)->pool->mem + BF(buf)->off); }
// A buffer or pool the CPU has no access to (NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS) maps to NULL, as in NVN, though the host's storage is CPU memory: EA SPORTS FC 27's Frostbite
// renderer takes a buffer's GPU address as GetAddress + (its data pointer − Map), with a null data pointer for such a buffer (OpcodeCopySubresource), so a real pointer made the
// copy's destination address 0 (FATAL buffer copy address is in no memory pool)
NVN(nvnBufferMap, void *, const NVNbuffer *buf) { return BF(buf)->pool->flags & NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS ? NULL : BF(buf)->pool->mem + BF(buf)->off; }
NVN(nvnBufferGetSize, size_t, const NVNbuffer *buf) { return (size_t)BF(buf)->size; }
NVN(nvnBufferGetMemoryPool, NVNmemoryPool *, const NVNbuffer *buf) { return (NVNmemoryPool *)BF(buf)->pool; }
NVN(nvnBufferGetMemoryOffset, ptrdiff_t, const NVNbuffer *buf) { return (ptrdiff_t)BF(buf)->off; }
NVN(nvnBufferFlushMappedRange, void, const NVNbuffer *buf, ptrdiff_t off, size_t n) { (void)buf; (void)off; (void)n; }        // unified memory: nothing to flush
NVN(nvnBufferInvalidateMappedRange, void, const NVNbuffer *buf, ptrdiff_t off, size_t n) { (void)buf; (void)off; (void)n; }

// Memory pool queries
NVN(nvnMemoryPoolGetFlags, NVNmemoryPoolFlags, const NVNmemoryPool *p) { return (NVNmemoryPoolFlags)((const pool_t *)p)->flags; }
NVN(nvnMemoryPoolMap, void *, const NVNmemoryPool *p) { return ((const pool_t *)p)->flags & NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS ? NULL : ((const pool_t *)p)->mem; }   // NULL without CPU access (nvnBufferMap)
NVN(nvnMemoryPoolGetSize, size_t, const NVNmemoryPool *p) { return (size_t)((const pool_t *)p)->size; }
NVN(nvnMemoryPoolInvalidateMappedRange, void, const NVNmemoryPool *p, ptrdiff_t off, size_t n) { (void)p; (void)off; (void)n; }
NVN(nvnMemoryPoolSetDebugLabel, void, NVNmemoryPool *p, const char *s) { (void)p; (void)s; }
// Debug labels: only for NVN debugging tools, unused by the host
NVN(nvnWindowSetDebugLabel, void, NVNwindow *w, const char *s) { (void)w; (void)s; }
NVN(nvnTexturePoolSetDebugLabel, void, NVNtexturePool *p, const char *s) { (void)p; (void)s; }
NVN(nvnSamplerPoolSetDebugLabel, void, NVNsamplerPool *p, const char *s) { (void)p; (void)s; }
NVN(nvnQueueSetDebugLabel, void, NVNqueue *q, const char *s) { (void)q; (void)s; }
NVN(nvnProgramSetDebugLabel, void, NVNprogram *p, const char *s) { (void)p; (void)s; }
NVN(nvnCommandBufferSetDebugLabel, void, NVNcommandBuffer *c, const char *s) { (void)c; (void)s; }
// Virtual pool (flags 0x200): a GPU address range with no storage; MapVirtual maps pages of physical pools into it. A request is 0x28 bytes: {physical pool, physical offset, virtual offset, size, storage class}.
typedef struct { NVNmemoryPool *phys; int64_t poff, voff; uint64_t size; int32_t storage_class, _pad; } map_req;
_Static_assert(sizeof(map_req) == 0x28, "NVNmappingRequest");
// True sparse virtual pools (Monster Hunter Rise: RE Engine's GPU heap is one 64 GB virtual pool, built from separately allocated 4 MB physical pools mapped at offsets of their own; boot maps 45 of them
// over the first 180 MB). A virtual pool larger than the whole application pool cannot be one alias of a physical pool, so at Initialize it gets an address range of its own (reserved, inaccessible),
// and every mapping request vm_remaps the physical pool's pages over its part of that range: the virtual addresses then read and write the same memory as the physical pool, as the GPU page table does
// on hardware, and buffer / texture addresses stay CPU addresses. Each run of mapped pages is registered with the replay as a pool of its own (mtl_pool_add), rebuilt when a request joins or splits runs.
// Unmapping puts inaccessible pages back. The reservation is the pool's size, at most 64 GB on the Mac and 1 GB on iOS.
// NVN maps at 4 KB, the host aliases at its 16 KB page: a request that is not 16 KB aligned (Monster Hunter Rise's ASTC 6x6 textures, whose mip tail starts 0x81000 into the texture),
// or one beyond the reservation, cannot be aliased. So every request is also kept in a page table at 4 KB (the physical memory behind each virtual page, and the mapping generation
// that set it), and the host reads textures of a sparse pool through it (nvn_sparse_copy): such a texture is uploaded from what the table points at, with unmapped pages read as 0,
// and uploaded again only when a mapping in its range changes or the CPU writes a physical page behind it (nvn_sparse_since; the physical pools are dirty-tracked). The game fills
// its sparse textures (flag 0x20, 32 GB into the pool) that way: it writes the tiles on the CPU into 4 MB physical pools, then maps them; no GPU copy writes them
// Every virtual pool gets that range at Initialize and keeps it until Finalize, since the game may create its textures and buffers, and take their addresses, before mapping anything,
// and may unmap or replace any part of a mapping later. OCTOPATH TRAVELER (Unreal Engine 4) gives each streamed texture a virtual pool of its own (0x20000 to 0x560000 bytes, flags
// 0x2a1), creates the texture in it, maps its mip tail from a 4 MB physical pool and later the higher mips from other pages of it, after writing their texels there on the CPU.
// The first mapping request decides how textures made from then on read the pool (vmode): one that maps the whole pool (Mario Kart World's streamed textures and its 250 MB pool of
// render targets, SMBW's 128 MB pool) makes it dense (1): its textures are ordinary ones over the range, which the CPU alias makes the physical pool's memory. Anything else, and any
// later request of a dense pool that does not map it whole again, makes it sparse (2): its textures read through the page table. A pool larger than the application pool is sparse
// from the start. Lichii: a dense pool used to give its range back and take the physical pool's address as its storage: addresses handed out before the mapping dangled, and a
// later unmap or a mapping of part of it to other physical pages was ignored or refused
enum { NVRUN = 1024, VPT_LEAF = 512 };   // a page-table leaf covers 2 MB
typedef struct { const uint8_t *phys; uint64_t gen; } vpte;
struct vpool { const pool_t *pool; uint8_t *base; uint64_t len, size, top; int nrun; struct { uint64_t lo, hi; } run[NVRUN]; vpte **pt; };   // len: reserved; size: the pool's; top: highest mapped end so far (logged per 128 MB)
static struct vpool **g_vp; static int g_nvp, g_cvp;   // the pools with a range of their own, under g_vp_mu
static pthread_mutex_t g_vp_mu = PTHREAD_MUTEX_INITIALIZER;
static _Atomic uint64_t g_vp_gen;   // bumped per mapping request
static struct vpool *vpool_of(const pool_t *o) { return o->vp; }
static uint8_t *vpool_reserve(pool_t *o) {   // → the reserved range, or NULL
    if (!(o->flags & 0x200)) return NULL;
#if PLAT_IOS   // 1 GB: Monster Hunter Rise's buffers stay under 0x20100000 through its opening, its textures (32 GB in) are read through the page table, and 4 GB guest + 1 GB stays near the 5 GB proven on a phone
    uint64_t len = o->size < (1ull << 30) ? o->size : 1ull << 30;
#else
    uint64_t len = o->size < (64ull << 30) ? o->size : 64ull << 30;
#endif
    len = (len + (uint64_t)getpagesize() - 1) & ~((uint64_t)getpagesize() - 1);
#if defined(_WIN32)   // Lichii: an alias region of the guest memory (nvn_direct_alias_pages maps into it there)
    void *a = nvn_direct_alias_reserve(len);
    if (!a) a = MAP_FAILED;
#else
    void *a = mmap(NULL, len, PROT_NONE, MAP_ANON | MAP_PRIVATE, -1, 0);
#endif
    if (a == MAP_FAILED) { LOG("[RED] virtual pool %p: cannot reserve %#llx bytes of address space (errno %d)", (const void *)o, (unsigned long long)len, errno); return NULL; }
    struct vpool *v = calloc(1, sizeof *v);
    if (!v || !(v->pt = calloc((o->size + (2u << 20) - 1) >> 21, sizeof *v->pt))) die("sparse pool allocation failed");
    v->pool = o; v->base = a; v->len = len; v->size = o->size;
    pthread_mutex_lock(&g_vp_mu);
    if (g_nvp == g_cvp && !(g_vp = realloc(g_vp, (g_cvp = g_cvp ? 2 * g_cvp : 64) * sizeof *g_vp))) die("sparse pool table allocation failed");
    int at = g_nvp; while (at > 0 && g_vp[at - 1]->base > v->base) at--;   // (kept sorted by base: vpool_at's binary search)
    memmove(&g_vp[at + 1], &g_vp[at], (size_t)(g_nvp - at) * sizeof *g_vp); g_vp[at] = v; g_nvp++; o->vp = v;
    o->vmode = o->size > DM_GAME_APP_POOL ? 2 : 0;
    pthread_mutex_unlock(&g_vp_mu);
    if (o->vmode == 2) LOG("[nvn] virtual pool %p (%#llx bytes, flags %#x): sparse, own address range %p+%#llx, physical pages are remapped into it", (const void *)o, (unsigned long long)o->size, o->flags, a, (unsigned long long)len);
    return a;
}
static void vpool_drop(pool_t *o) {   // caller holds g_vp_mu
    struct vpool *v = o->vp; if (!v) return;
    for (int k = 0; k < v->nrun; k++) mtl_pool_del(v->base + v->run[k].lo);
#if !PLAT_APPLE
    mtl_pool_alias(v->base, v->len < v->size ? v->len : v->size, NULL);   // (the GPU's aliases go with the range: a pool reserved there later starts unmapped)
#endif
#if defined(_WIN32)
    nvn_direct_alias_release(v->base);
#else
    munmap(v->base, v->len);
#endif
    for (uint64_t l = 0; l < (v->size + (2u << 20) - 1) >> 21; l++) free(v->pt[l]);
    for (int i = 0; i < g_nvp; i++) if (g_vp[i] == v) { memmove(&g_vp[i], &g_vp[i + 1], (size_t)(g_nvp - 1 - i) * sizeof *g_vp); g_nvp--; break; }
    free(v->pt); free(v); o->vp = NULL;
}
static void vpool_release(pool_t *o) { pthread_mutex_lock(&g_vp_mu); vpool_drop(o); pthread_mutex_unlock(&g_vp_mu); }
// Mapped runs of a sparse pool: [lo, hi) sorted, disjoint, adjacent runs merged. Caller holds g_vp_mu
static void vrun_set(struct vpool *v, uint64_t lo, uint64_t hi, int mapped) {
    uint64_t nl[NVRUN][2]; int nn = 0; uint64_t mlo = lo, mhi = hi;
    for (int k = 0; k < v->nrun; k++) {
        uint64_t a = v->run[k].lo, b = v->run[k].hi;
        if (b < lo || a > hi || (!mapped && (b == lo || a == hi))) { if (nn < NVRUN) { nl[nn][0] = a; nl[nn][1] = b; nn++; } continue; }   // untouched (an unmap leaves adjacent runs alone)
        mtl_pool_del(v->base + a);
        if (mapped) { if (a < mlo) mlo = a; if (b > mhi) mhi = b; continue; }   // joins the new run
        if (a < lo && nn < NVRUN) { nl[nn][0] = a; nl[nn][1] = lo; nn++; mtl_pool_add(v->base + a, lo - a); }   // what the unmap leaves on either side
        if (b > hi && nn < NVRUN) { nl[nn][0] = hi; nl[nn][1] = b; nn++; mtl_pool_add(v->base + hi, b - hi); }
    }
    if (mapped && nn < NVRUN) { nl[nn][0] = mlo; nl[nn][1] = mhi; nn++; mtl_pool_add(v->base + mlo, mhi - mlo); }
    if (nn == NVRUN) LOG("[RED] virtual pool %p: more than %d separate mapped runs", (const void *)v->pool, NVRUN);
    for (int a = 1; a < nn; a++) for (int b = a; b > 0 && nl[b][0] < nl[b - 1][0]; b--) { uint64_t t0 = nl[b][0], t1 = nl[b][1]; nl[b][0] = nl[b - 1][0]; nl[b][1] = nl[b - 1][1]; nl[b - 1][0] = t0; nl[b - 1][1] = t1; }
    v->nrun = nn; for (int k = 0; k < nn; k++) { v->run[k].lo = nl[k][0]; v->run[k].hi = nl[k][1]; }
}
// Page table of a sparse pool (see above). Caller holds g_vp_mu
static const vpte *vpt_get(const struct vpool *v, uint64_t o) { const vpte *l = v->pt[o >> 21]; return l ? &l[(o >> 12) & (VPT_LEAF - 1)] : NULL; }
static void vpt_set(struct vpool *v, uint64_t voff, uint64_t size, const uint8_t *phys) {
    uint64_t gen = atomic_fetch_add(&g_vp_gen, 1) + 1;
    for (uint64_t o = 0; o < size; o += 0x1000) {
        vpte **l = &v->pt[(voff + o) >> 21];
        if (!*l && !phys) continue;   // never mapped: stays unmapped
        if (!*l && !(*l = calloc(VPT_LEAF, sizeof **l))) die("sparse pool page table allocation failed");
        (*l)[((voff + o) >> 12) & (VPT_LEAF - 1)] = (vpte){phys ? phys + o : NULL, gen};
    }
}
static int vpool_map(struct vpool *v, const map_req *r) {   // caller holds g_vp_mu; → 0 on failure
    uint64_t pg = (uint64_t)getpagesize();
    if (r->voff < 0 || (uint64_t)r->voff + r->size > v->size) { LOG("[RED] virtual pool %p: request at %#llx+%#llx is beyond the pool's %#llx", (const void *)v->pool, (unsigned long long)r->voff, (unsigned long long)r->size, (unsigned long long)v->size); return 0; }
    if (((uint64_t)r->voff | r->size | (uint64_t)r->poff) & 0xFFF) { LOG("[RED] virtual pool %p: request %#llx+%#llx (physical offset %#llx) is not 4 KB aligned", (const void *)v->pool, (unsigned long long)r->voff, (unsigned long long)r->size, (unsigned long long)r->poff); return 0; }
    const pool_t *ph = (const pool_t *)r->phys;
    if (ph && (!ph->mem || r->poff < 0 || (uint64_t)r->poff + r->size > ph->size)) { LOG("[RED] virtual pool %p: physical pool %p (%#llx bytes, storage %p) cannot supply %#llx+%#llx", (const void *)v->pool, (const void *)ph, (unsigned long long)ph->size, (void *)ph->mem, (unsigned long long)r->poff, (unsigned long long)r->size); return 0; }
    vpt_set(v, (uint64_t)r->voff, r->size, ph ? ph->mem + r->poff : NULL);
    uintptr_t to = (uintptr_t)(v->base + r->voff);
    // Not aliased (see above): only textures read through the page table see it; the GPU cannot address it directly
    if ((uint64_t)r->voff + r->size > v->len || ((uintptr_t)to | r->size | (uint64_t)r->voff) & (pg - 1) || (ph && (((uintptr_t)ph->mem + (uintptr_t)r->poff) & (pg - 1)))) {
        static _Atomic uint64_t n; if (!atomic_fetch_add(&n, 1)) LOG("[nvn] virtual pool %p: a request %s (%s %#llx+%#llx) is kept in the page table only: textures read it, the GPU cannot address it (later cases are not logged)",
                                                                   (const void *)v->pool, (uint64_t)r->voff + r->size > v->len ? "beyond the reservation" : "not aligned to the host page", r->phys ? "map" : "unmap", (unsigned long long)r->voff, (unsigned long long)r->size);
        return 1;
    }
    if (!r->phys) {   // unmap: inaccessible pages again
#if !PLAT_APPLE
        mtl_pool_alias((void *)to, r->size, NULL);
#endif
#if defined(_WIN32)
        nvn_direct_alias_unmap((void *)to, r->size);
#else
        if (mmap((void *)to, r->size, PROT_NONE, MAP_ANON | MAP_PRIVATE | MAP_FIXED, -1, 0) == MAP_FAILED) { LOG("[RED] virtual pool unmap %#lx+%#llx failed (errno %d)", (unsigned long)to, (unsigned long long)r->size, errno); return 0; }
#endif
        vrun_set(v, (uint64_t)r->voff, (uint64_t)r->voff + r->size, 0); return 1;
    }
    uintptr_t from = (uintptr_t)ph->mem + (uintptr_t)r->poff;
#if !PLAT_APPLE
    // Android: the physical pages are placed Vulkan memory, which cannot be remapped here. The GPU's lookups of the range answer with the physical pool's memory
    // (mtl_pool_alias); the game has no CPU access to a virtual pool, and the host reads its textures through the page table
    mtl_pool_alias((void *)to, r->size, (const void *)from);
    // Lichii: the physical pages are guest memory, so the range maps them for the CPU too, as vm_remap does below: the replay reads and writes buffers of a virtual pool
    // (indirect draw arguments, report values) at their addresses
    if (nvn_direct_alias_pages((void *)to, (uint64_t)from, r->size)) {
        static _Atomic uint64_t n; if (!atomic_fetch_add(&n, 1)) LOG("[RED] virtual pool %p: %#llx+%#llx: physical pages with no guest memory behind them stay inaccessible to the CPU (later cases are not logged)",
                                                                   (const void *)v->pool, (unsigned long long)r->voff, (unsigned long long)r->size);
    }
    return 1;
#else
    vm_prot_t cur, max;
    kern_return_t kr = vm_remap(mach_task_self(), &to, r->size, 0, VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE, mach_task_self(), (vm_address_t)from, FALSE, &cur, &max, VM_INHERIT_NONE);
    if (kr != KERN_SUCCESS || to != (vm_address_t)(v->base + r->voff)) { LOG("[RED] virtual pool remap %#lx → %p (%#llx bytes) failed: %d", (unsigned long)from, (void *)(v->base + r->voff), (unsigned long long)r->size, kr); return 0; }
    mprotect((void *)to, r->size, PROT_READ | PROT_WRITE);   // the source may be write-protected by the dirty tracker at this moment; the alias itself is plain memory
    vrun_set(v, (uint64_t)r->voff, (uint64_t)r->voff + r->size, 1);
    uint64_t end = (uint64_t)r->voff + r->size;
    if (end >> 27 > v->top >> 27) LOG("[nvn] virtual pool %p: mapped up to %#llx (%d runs)", (const void *)v->pool, (unsigned long long)end, v->nrun);
    if (end > v->top) v->top = end;
    return 1;
#endif
}
NVN(nvnMemoryPoolMapVirtual, NVNboolean, NVNmemoryPool *p, int n, const NVNmappingRequest *rq) {
    pool_t *o = (pool_t *)p; const map_req *r = (const map_req *)rq;
    int unmap = 0; for (int i = 0; i < n; i++) unmap |= !r[i].phys;
    if (unmap) nvn_pool_unmap(1);   // (a reader of the pool's storage that found a page mapped reads it before it goes: nvn_pool_use)
    pthread_mutex_lock(&g_vp_mu);
    struct vpool *v = vpool_of(o); int ok = v != NULL;
    if (v && n > 0 && o->vmode != 2) {   // the first request decides; a dense pool stays one only while its requests map it whole (see the sparse pools above)
        int whole = n == 1 && r[0].phys && r[0].voff == 0 && r[0].size == o->size;
        if (o->vmode == 1 && !whole) { static _Atomic int nl; if (atomic_fetch_add(&nl, 1) < 8) LOG("[nvn] virtual pool %p (%#llx bytes) was mapped whole and is now mapped in part: sparse from now on (its textures made before keep reading its range)", (void *)o, (unsigned long long)o->size); }
        o->vmode = whole && o->vmode == 0 ? 1 : whole ? o->vmode : 2;
    }
    for (int i = 0; v && i < n && ok; i++) ok = vpool_map(v, &r[i]);
    pthread_mutex_unlock(&g_vp_mu);
    if (unmap) nvn_pool_unmap(0);
    for (int i = 0; i < n; i++) if (r[i].phys) pc_use((const pool_t *)r[i].phys, 'm', r[i].poff, r[i].size);   // (requested: counted whether or not they succeed)
    if (!v) LOG("[RED] MapVirtual: virtual pool %p (flags %#x) has no address range of its own: nothing is mapped", (void *)o, o->flags);
    return (NVNboolean)ok;
}

// Samplers: the getters read the parameters in the object. Reduction (min / max of the texels a filtered sample covers) maps to the Metal 26 sampler reduction mode (nvn_mtl.m samp_by_id)
NVN(nvnSamplerBuilderSetReductionFilter, void, NVNsamplerBuilder *b, NVNsamplerReduction r) { SB(b)->reduction = (uint8_t)r; }
NVN(nvnSamplerSetDebugLabel, void, NVNsampler *s, const char *l) { (void)s; (void)l; }
#define SP(s) (&((const samp_t *)(s))->p)
NVN(nvnSamplerGetMinMagFilter, void, const NVNsampler *s, NVNminFilter *mi, NVNmagFilter *ma) { *mi = (NVNminFilter)SP(s)->minf; *ma = (NVNmagFilter)SP(s)->magf; }
NVN(nvnSamplerGetWrapMode, void, const NVNsampler *s, NVNwrapMode *a, NVNwrapMode *b, NVNwrapMode *c) { *a = (NVNwrapMode)SP(s)->ws; *b = (NVNwrapMode)SP(s)->wt; *c = (NVNwrapMode)SP(s)->wr; }
NVN(nvnSamplerGetLodClamp, void, const NVNsampler *s, float *lo, float *hi) { *lo = SP(s)->lod_min; *hi = SP(s)->lod_max; }
NVN(nvnSamplerGetLodBias, float, const NVNsampler *s) { return SP(s)->lod_bias; }
NVN(nvnSamplerGetCompare, void, const NVNsampler *s, NVNcompareMode *m, NVNcompareFunc *f) { *m = (NVNcompareMode)SP(s)->cmp_mode; *f = (NVNcompareFunc)SP(s)->cmp_func; }
NVN(nvnSamplerGetBorderColor, void, const NVNsampler *s, float *c) { memcpy(c, SP(s)->border, 16); }
NVN(nvnSamplerGetMaxAnisotropy, float, const NVNsampler *s) { return SP(s)->aniso; }
NVN(nvnSamplerGetReductionFilter, NVNsamplerReduction, const NVNsampler *s) { return (NVNsamplerReduction)SP(s)->reduction; }
// whether two samplers have the same state (games use it to deduplicate sampler pool entries)
NVN(nvnSamplerCompare, NVNboolean, const NVNsampler *a, const NVNsampler *b) {
    const samp_par *x = SP(a), *y = SP(b);
    return x->minf == y->minf && x->magf == y->magf && x->ws == y->ws && x->wt == y->wt && x->wr == y->wr && x->cmp_mode == y->cmp_mode && x->cmp_func == y->cmp_func && x->reduction == y->reduction && x->border_int == y->border_int &&
           x->lod_min == y->lod_min && x->lod_max == y->lod_max && x->lod_bias == y->lod_bias && x->aniso == y->aniso && !memcmp(x->border, y->border, sizeof x->border);
}

// Texture builder / texture / view
NVN(nvnTextureBuilderSetSize2D, void, NVNtextureBuilder *b, int w, int h) { TB(b)->d.width = w; TB(b)->d.height = h; TB(b)->d.depth = 1; }
NVN(nvnTextureBuilderSetSize3D, void, NVNtextureBuilder *b, int w, int h, int d) { TB(b)->d.width = w; TB(b)->d.height = h; TB(b)->d.depth = d; }
NVN(nvnTextureBuilderSetSwizzle, void, NVNtextureBuilder *b, NVNtextureSwizzle r, NVNtextureSwizzle g, NVNtextureSwizzle bl, NVNtextureSwizzle a) {
    tex_builder_t *o = (tex_builder_t *)b; o->set_swz = 1; o->swz = (uint16_t)((r & 7) | (g & 7) << 3 | (bl & 7) << 6 | (a & 7) << 9);
    if (r != NVN_TEXTURE_SWIZZLE_R || g != NVN_TEXTURE_SWIZZLE_G || bl != NVN_TEXTURE_SWIZZLE_B || a != NVN_TEXTURE_SWIZZLE_A) ONCE_LOG("TextureBuilderSetSwizzle not identity (%d %d %d %d): remapped with a Metal view when sampling", (int)r, (int)g, (int)bl, (int)a);
}
// STENCIL: the texture reads the stencil of its depth-stencil storage, as a stencil-mode view does (tex_by_id in nvn_mtl.m). Unreal Engine 4.27 makes its stencil shader resources this way: a second texture
// on the depth buffer's storage, built in stencil mode (DRAGON QUEST VII Reimagined's character occlusion mask, edbfc7a088be545e, tests stencil bits 0x1 and 0x84 of two of them; read as depth, the
// bits followed the depth values and the hero was drawn as a dithered silhouette inside a dithered square)
NVN(nvnTextureBuilderSetDepthStencilMode, void, NVNtextureBuilder *b, NVNtextureDepthStencilMode m) { TB(b)->stencil = m == NVN_TEXTURE_DEPTH_STENCIL_MODE_STENCIL; }
NVN(nvnTextureBuilderGetStorageClass, NVNstorageClass, const NVNtextureBuilder *b) { (void)b; return 0; }
// Used by Bowser's Fury: the texture's channel swizzle (set by TextureBuilderSetSwizzle at build time, identity R G B A when unset) and storage class (the host has none, 0 as for the builder)
NVN(nvnTextureGetSwizzle, void, const NVNtexture *t, NVNtextureSwizzle *r, NVNtextureSwizzle *g, NVNtextureSwizzle *b, NVNtextureSwizzle *a) {
    unsigned w = tex_swz(CTX(t));
    if (r) *r = (NVNtextureSwizzle)(w & 7); if (g) *g = (NVNtextureSwizzle)(w >> 3 & 7); if (b) *b = (NVNtextureSwizzle)(w >> 6 & 7); if (a) *a = (NVNtextureSwizzle)(w >> 9 & 7);
}
NVN(nvnTextureGetStorageClass, NVNstorageClass, const NVNtexture *t) { (void)t; return 0; }
NVN(nvnTextureBuilderGetZCullStorageSize, size_t, const NVNtextureBuilder *b) { return zcull_size(&TB(b)->d); }   // as for the texture (zcull_size)
// Packaged texture data in CPU memory (not written into the pool by the game): NVN copies it into the texture storage at TextureInitialize (the CPU cannot write a compressible pool; the driver copies). 13 Sentinels creates all its textures this way
NVN(nvnTextureBuilderSetPackagedTextureData, void, NVNtextureBuilder *b, const void *data) { TB(b)->pkg_data = data; }
NVN(nvnTextureGetFormat, NVNformat, const NVNtexture *t) { return (NVNformat)CTX(t)->d.format; }
NVN(nvnTextureGetTarget, NVNtextureTarget, const NVNtexture *t) { return (NVNtextureTarget)CTX(t)->d.target; }
NVN(nvnTextureSetDebugLabel, void, NVNtexture *t, const char *s) { (void)t; (void)s; }
NVN(nvnTextureViewGetLevels, NVNboolean, const NVNtextureView *v, int *base, int *n) { const view_t *w = (const view_t *)v; *base = w->base_level; *n = w->nlevels; return w->set_levels; }
NVN(nvnTextureViewGetTarget, NVNboolean, const NVNtextureView *v, NVNtextureTarget *t) { const view_t *w = (const view_t *)v; *t = (NVNtextureTarget)w->target; return w->set_target; }
NVN(nvnTextureViewSetDepthStencilMode, void, NVNtextureView *v, NVNtextureDepthStencilMode m) { VW(v)->set_dsmode = 1; VW(v)->dsmode = (uint8_t)m; }   // STENCIL: the replay binds a stencil-only view (tex_by_id)
NVN(nvnTextureViewSetSwizzle, void, NVNtextureView *v, NVNtextureSwizzle r, NVNtextureSwizzle g, NVNtextureSwizzle bl, NVNtextureSwizzle a) {
    view_t *w = VW(v); w->set_swizzle = 1; w->swz[0] = (uint8_t)r; w->swz[1] = (uint8_t)g; w->swz[2] = (uint8_t)bl; w->swz[3] = (uint8_t)a;
    if (r != NVN_TEXTURE_SWIZZLE_R || g != NVN_TEXTURE_SWIZZLE_G || bl != NVN_TEXTURE_SWIZZLE_B || a != NVN_TEXTURE_SWIZZLE_A) ONCE_LOG("TextureViewSetSwizzle not identity (%d %d %d %d): remapped with a Metal view when sampling", (int)r, (int)g, (int)bl, (int)a);
}

// Color state: alpha test function, logic op (the fields already exist in color_st)
NVN(nvnColorStateSetAlphaTest, void, NVNcolorState *s, NVNalphaFunc f) { ST(color_st, s)->alpha_func = (uint32_t)f; }
NVN(nvnColorStateSetLogicOp, void, NVNcolorState *s, NVNlogicOp op) { ST(color_st, s)->logic_op = (uint32_t)op; }
// Getters and the rest of the multisample state: Pokémon Legends: Z-A (Trinity, SDK 19) reads its state objects and builders back when it builds its pipeline states; the stub returned
// whatever was left in x0 (the object's address), so a blend enable or depth test read back as true, a texture builder's width as an address
NVN(nvnColorStateGetBlendEnable, NVNboolean, const NVNcolorState *s, int i) { return i >= 0 && i < 8 && (((const color_st *)s)->blend_en >> i & 1); }
NVN(nvnColorStateGetAlphaTest, NVNalphaFunc, const NVNcolorState *s) { return (NVNalphaFunc)((const color_st *)s)->alpha_func; }
NVN(nvnColorStateGetLogicOp, NVNlogicOp, const NVNcolorState *s) { return (NVNlogicOp)((const color_st *)s)->logic_op; }
NVN(nvnChannelMaskStateGetChannelMask, void, const NVNchannelMaskState *s, int i, NVNboolean *r, NVNboolean *g, NVNboolean *b, NVNboolean *a) {
    uint32_t m = i >= 0 && i < 8 ? ((const chmask_st *)s)->mask >> (4 * i) & 0xf : 0;
    if (r) *r = (m & 1) != 0; if (g) *g = (m & 2) != 0; if (b) *b = (m & 4) != 0; if (a) *a = (m & 8) != 0;
}
NVN(nvnBlendStateGetBlendTarget, int, const NVNblendState *s) { return (int)((const blend_st *)s)->target; }
NVN(nvnBlendStateGetBlendFunc, void, const NVNblendState *s, NVNblendFunc *sr, NVNblendFunc *dr, NVNblendFunc *sa, NVNblendFunc *da) {
    const blend_st *o = (const blend_st *)s;
    if (sr) *sr = (NVNblendFunc)o->src_rgb; if (dr) *dr = (NVNblendFunc)o->dst_rgb; if (sa) *sa = (NVNblendFunc)o->src_a; if (da) *da = (NVNblendFunc)o->dst_a;
}
NVN(nvnBlendStateGetBlendEquation, void, const NVNblendState *s, NVNblendEquation *er, NVNblendEquation *ea) {
    const blend_st *o = (const blend_st *)s; if (er) *er = (NVNblendEquation)o->eq_rgb; if (ea) *ea = (NVNblendEquation)o->eq_a;
}
NVN(nvnBlendStateGetAdvancedMode, NVNblendAdvancedMode, const NVNblendState *s) { return (NVNblendAdvancedMode)((const blend_st *)s)->adv; }
NVN(nvnBlendStateGetAdvancedPremultipliedSrc, NVNboolean, const NVNblendState *s) { return !((const blend_st *)s)->adv_npm; }
NVN(nvnBlendStateGetAdvancedOverlap, NVNblendAdvancedOverlap, const NVNblendState *s) { return (NVNblendAdvancedOverlap)((const blend_st *)s)->adv_ovl; }
NVN(nvnPolygonStateGetCullFace, NVNface, const NVNpolygonState *s) { return (NVNface)((const poly_st *)s)->cull; }
NVN(nvnPolygonStateGetFrontFace, NVNfrontFace, const NVNpolygonState *s) { return (NVNfrontFace)((const poly_st *)s)->front; }
NVN(nvnPolygonStateGetPolygonMode, NVNpolygonMode, const NVNpolygonState *s) { return (NVNpolygonMode)((const poly_st *)s)->mode; }
NVN(nvnPolygonStateGetPolygonOffsetEnables, NVNpolygonOffsetEnable, const NVNpolygonState *s) { return (NVNpolygonOffsetEnable)((const poly_st *)s)->offset_en; }
NVN(nvnDepthStencilStateGetDepthTestEnable, NVNboolean, const NVNdepthStencilState *s) { return ((const ds_st *)s)->dt; }
NVN(nvnDepthStencilStateGetDepthWriteEnable, NVNboolean, const NVNdepthStencilState *s) { return ((const ds_st *)s)->dw; }
NVN(nvnDepthStencilStateGetDepthFunc, NVNdepthFunc, const NVNdepthStencilState *s) { return (NVNdepthFunc)((const ds_st *)s)->df; }
NVN(nvnDepthStencilStateGetStencilTestEnable, NVNboolean, const NVNdepthStencilState *s) { return ((const ds_st *)s)->st; }
NVN(nvnDepthStencilStateGetStencilFunc, NVNstencilFunc, const NVNdepthStencilState *s, NVNface f) { const ds_st *o = (const ds_st *)s; return (NVNstencilFunc)(f == NVN_FACE_BACK ? o->sf_b : o->sf_f); }
NVN(nvnDepthStencilStateGetStencilOp, void, const NVNdepthStencilState *s, NVNface f, NVNstencilOp *fail, NVNstencilOp *zfail, NVNstencilOp *zpass) {
    const ds_st *o = (const ds_st *)s; int b = f == NVN_FACE_BACK;
    if (fail) *fail = (NVNstencilOp)(b ? o->fail_b : o->fail_f); if (zfail) *zfail = (NVNstencilOp)(b ? o->zfail_b : o->zfail_f); if (zpass) *zpass = (NVNstencilOp)(b ? o->zpass_b : o->zpass_f);
}
// Multisample state: recorded only (the replay uses the sample count, multisample enable and alpha to coverage)
NVN(nvnMultisampleStateSetAlphaToCoverageDither, void, NVNmultisampleState *s, uint32_t e) { ST(ms_st, s)->a2c_dither = B(e) != 0; }
NVN(nvnMultisampleStateSetRasterSamples, void, NVNmultisampleState *s, int n) { ST(ms_st, s)->raster_samples = (uint8_t)n; }
NVN(nvnMultisampleStateSetCoverageModulationMode, void, NVNmultisampleState *s, NVNcoverageModulationMode m) { ST(ms_st, s)->cov_mod = (uint8_t)m; }
NVN(nvnMultisampleStateSetCoverageToColorEnable, void, NVNmultisampleState *s, uint32_t e) { ST(ms_st, s)->cov2color_en = B(e) != 0; }
NVN(nvnMultisampleStateSetCoverageToColorOutput, void, NVNmultisampleState *s, int i) { ST(ms_st, s)->cov2color_out = (uint8_t)i; }
NVN(nvnMultisampleStateSetSampleLocationsGridEnable, void, NVNmultisampleState *s, uint32_t e) { ST(ms_st, s)->sl_grid = B(e) != 0; }
NVN(nvnMultisampleStateGetAlphaToCoverageEnable, NVNboolean, const NVNmultisampleState *s) { return ((const ms_st *)s)->a2c; }
NVN(nvnMultisampleStateGetAlphaToCoverageDither, NVNboolean, const NVNmultisampleState *s) { return ((const ms_st *)s)->a2c_dither; }
NVN(nvnMultisampleStateGetRasterSamples, int, NVNmultisampleState *s) { return ((const ms_st *)s)->raster_samples; }
NVN(nvnMultisampleStateGetCoverageModulationMode, NVNcoverageModulationMode, const NVNmultisampleState *s) { return (NVNcoverageModulationMode)((const ms_st *)s)->cov_mod; }
NVN(nvnMultisampleStateGetCoverageToColorEnable, NVNboolean, const NVNmultisampleState *s) { return ((const ms_st *)s)->cov2color_en; }
NVN(nvnMultisampleStateGetCoverageToColorOutput, int, const NVNmultisampleState *s) { return ((const ms_st *)s)->cov2color_out; }
NVN(nvnMultisampleStateGetSampleLocationsEnable, NVNboolean, const NVNmultisampleState *s) { return ((const ms_st *)s)->sl_en; }
NVN(nvnMultisampleStateGetSampleLocationsGridEnable, NVNboolean, const NVNmultisampleState *s) { return ((const ms_st *)s)->sl_grid; }
// Builder and pool queries
NVN(nvnBufferBuilderGetSize, size_t, const NVNbufferBuilder *b) { return (size_t)((const buf_builder_t *)b)->size; }
NVN(nvnBufferBuilderGetMemoryOffset, ptrdiff_t, const NVNbufferBuilder *b) { return (ptrdiff_t)((const buf_builder_t *)b)->off; }
// The storage set with SetStorage (EA SPORTS FC 27's Frostbite renderer reads it back from its texture builders before it initializes the textures). The pool getters return the
// pool's address, as nvnTextureGetMemoryPool does; the header the prototypes come from declares the builders' by value, so they get no signature check
NVN_NOPROTO(nvnBufferBuilderGetMemoryPool, NVNmemoryPool *, const NVNbufferBuilder *b) { return (NVNmemoryPool *)((const buf_builder_t *)b)->pool; }
NVN_NOPROTO(nvnTextureBuilderGetMemoryPool, NVNmemoryPool *, const NVNtextureBuilder *b) { return (NVNmemoryPool *)((const tex_builder_t *)b)->pool; }
NVN(nvnTextureBuilderGetMemoryOffset, ptrdiff_t, const NVNtextureBuilder *b) { return (ptrdiff_t)((const tex_builder_t *)b)->off; }
NVN_NOPROTO(nvnEventBuilderGetMemoryPool, NVNmemoryPool *, const NVNeventBuilder *b) { return (NVNmemoryPool *)((const ev_builder_t *)b)->pool; }
NVN_NOPROTO(nvnEventBuilderGetMemoryOffset, int64_t, const NVNeventBuilder *b) { return ((const ev_builder_t *)b)->off; }
NVN(nvnTextureBuilderGetTarget, NVNtextureTarget, const NVNtextureBuilder *b) { return (NVNtextureTarget)((const tex_builder_t *)b)->d.target; }
NVN(nvnTextureBuilderGetWidth, int, const NVNtextureBuilder *b) { return (int)((const tex_builder_t *)b)->d.width; }
NVN(nvnTextureBuilderGetHeight, int, const NVNtextureBuilder *b) { return (int)((const tex_builder_t *)b)->d.height; }
NVN(nvnTextureBuilderGetDepth, int, const NVNtextureBuilder *b) { return (int)((const tex_builder_t *)b)->d.depth; }
NVN(nvnTextureBuilderGetLevels, int, const NVNtextureBuilder *b) { return (int)((const tex_builder_t *)b)->d.levels; }
NVN(nvnTextureBuilderGetFormat, NVNformat, const NVNtextureBuilder *b) { return (NVNformat)((const tex_builder_t *)b)->d.format; }
NVN(nvnTextureBuilderGetSamples, int, const NVNtextureBuilder *b) { return ((const tex_builder_t *)b)->d.samples; }
NVN(nvnTextureBuilderGetStride, ptrdiff_t, const NVNtextureBuilder *b) { return (ptrdiff_t)((const tex_builder_t *)b)->d.stride; }
NVN(nvnTextureViewGetFormat, NVNboolean, const NVNtextureView *v, NVNformat *f) { const view_t *w = (const view_t *)v; if (f) *f = (NVNformat)w->format; return w->set_format; }
NVN(nvnTextureViewGetDepthStencilMode, NVNboolean, const NVNtextureView *v, NVNtextureDepthStencilMode *m) { const view_t *w = (const view_t *)v; if (m) *m = (NVNtextureDepthStencilMode)w->dsmode; return w->set_dsmode; }
NVN(nvnTextureViewGetSwizzle, NVNboolean, const NVNtextureView *v, NVNtextureSwizzle *r, NVNtextureSwizzle *g, NVNtextureSwizzle *b, NVNtextureSwizzle *a) {
    const view_t *w = (const view_t *)v; NVNtextureSwizzle *o[4] = {r, g, b, a};
    for (int c = 0; c < 4; c++) if (o[c]) *o[c] = (NVNtextureSwizzle)(w->set_swizzle ? w->swz[c] : NVN_TEXTURE_SWIZZLE_R + c);
    return w->set_swizzle;
}
NVN(nvnTextureGetDepthStencilMode, NVNtextureDepthStencilMode, const NVNtexture *t) { return ((const tex_t *)t)->stencil ? NVN_TEXTURE_DEPTH_STENCIL_MODE_STENCIL : NVN_TEXTURE_DEPTH_STENCIL_MODE_DEPTH; }
// Queue priority: kept for the getter (the replay gives every queue the same priority)
NVN_NOPROTO(nvnQueueBuilderSetQueuePriority, void, NVNqueueBuilder *b, int prio) { QB(b)->prio = prio; }
// The driver's external graphics debugger switch (Nsight / NVN debugger attach): Super Smash Bros. Ultimate fetches it with nvnDeviceGetProcAddress(NULL, …) before creating its device and calls
// it without a null check (main+0x37fe92c); the host has no debugger to attach
NVN_NOPROTO(nvnSetExternalDebuggerEnabledNVX, void, NVNboolean enable) { (void)enable; }
// Transform feedback (vertex outputs captured into buffers): Pokémon Legends: Z-A skins its characters this way. A vertex-only program writes skinned positions and packed normals / tangents
// under rasterizer discard, and the character draws read those buffers as vertex streams. Recorded here and run at replay by the shader's main0_tf entry point, a vertex function that stores the
// captured outputs into the bound buffers (dmsh_build.py vs_tf, the layout from the program's control section; nvn_mtl.m tf_* for the replay)
NVN(nvnCommandBufferBindTransformFeedbackBuffer, void, NVNcommandBuffer *c, int i, NVNbufferAddress a, size_t n) { *REC(c, OP_TF_BIND, p_tf_bind) = (p_tf_bind){i, 0, a, n}; }
NVN(nvnCommandBufferBindTransformFeedbackBuffers, void, NVNcommandBuffer *c, int first, int n, const NVNbufferRange *r) {
    for (int i = 0; i < n; i++) *REC(c, OP_TF_BIND, p_tf_bind) = (p_tf_bind){first + i, 0, r[i].address, r[i].size};
}
NVN(nvnCommandBufferBeginTransformFeedback, void, NVNcommandBuffer *c, NVNbufferAddress a) { *REC(c, OP_TF_STATE, p_tf_state) = (p_tf_state){0, 0, a}; }
NVN(nvnCommandBufferEndTransformFeedback, void, NVNcommandBuffer *c, NVNbufferAddress a) { *REC(c, OP_TF_STATE, p_tf_state) = (p_tf_state){1, 0, a}; }
NVN(nvnCommandBufferPauseTransformFeedback, void, NVNcommandBuffer *c, NVNbufferAddress a) { *REC(c, OP_TF_STATE, p_tf_state) = (p_tf_state){2, 0, a}; }
NVN(nvnCommandBufferResumeTransformFeedback, void, NVNcommandBuffer *c, NVNbufferAddress a) { *REC(c, OP_TF_STATE, p_tf_state) = (p_tf_state){3, 0, a}; }
NVN(nvnCommandBufferDrawTransformFeedback, void, NVNcommandBuffer *c, NVNdrawPrimitive m, NVNbufferAddress a) {
    *REC(c, OP_DRAW, p_draw) = (p_draw){.prim=m,.instances=1,.ind=a+16};
}

// Device: combined texture handle = texture id | sampler id << 20 (same encoding as the separate handles)
NVN(nvnDeviceGetTextureHandle, NVNtextureHandle, const NVNdevice *d, int tex, int samp) { (void)d; return (NVNtextureHandle)((uint64_t)(uint32_t)tex | (uint64_t)(uint32_t)samp << 20); }
// Handle for texel fetches only (Monster Hunter Rise): a fetch reads the texture descriptor and no sampler, so the handle is the texture ID with sampler ID 0
NVN(nvnDeviceGetTexelFetchHandle, NVNtextureHandle, const NVNdevice *d, int tex) { (void)d; return (NVNtextureHandle)(uint64_t)(uint32_t)tex; }
// Convert the raw 614.4 MHz report to ns on the same guest clock as GetCurrentTimestampInNanoseconds.
NVN(nvnDeviceGetTimestampInNanoseconds, uint64_t, const NVNdevice *d, const NVNcounterData *c) { (void)d; return nvn_timestamp_to_ns(c->timestamp); }
// Current GPU time (ns): the same clock as the counter timestamps -- the guest clock (guest_ns, nvn_mtl.m gpu_ticks)
NVN(nvnDeviceGetCurrentTimestampInNanoseconds, uint64_t, const NVNdevice *d) { (void)d; return guest_ns(); }
NVN(nvnDeviceRegisterFastClearColor, NVNboolean, NVNdevice *d, const float *c, NVNformat f) { (void)d; (void)c; (void)f; return 1; }
// Same as the color one: a fast clear value is only a hint to the Tegra hardware (Tears of the Kingdom registers its depth clear at startup)
NVN(nvnDeviceRegisterFastClearDepth, NVNboolean, NVNdevice *d, float v) { (void)d; (void)v; return 1; }
// The size of the driver's cache for shaders it compiles from intermediate code (The Great Ace Attorney Chronicles sets 10 at startup); the host compiles nothing at run time
NVN(nvnDeviceSetIntermediateShaderCache, void, NVNdevice *d, int n) { (void)d; (void)n; }

// Window: number of active swapchain textures (0 at the builder: all of them). Kept for the getters; the host rotates through all of the window's textures
NVN_NOPROTO(nvnWindowBuilderSetNumActiveTextures, void, NVNwindowBuilder *b, int n) { ((win_builder_t *)b)->active = n; }
NVN_NOPROTO(nvnWindowSetNumActiveTextures, void, NVNwindow *w, int n) {
    if (((win_t *)w)->active != n) LOG("[nvn] the game sets %d active window textures of %d (the host rotates through all of them)", n, ((win_t *)w)->ntex);
    ((win_t *)w)->active = n;
}

// Command buffer memory usage: record bytes written in this recording (accumulated across segments; the host does not use control memory, always 0); the memory callback switch is ignored.
// It used to be cur - begin: once a recording overflowed into a host segment, cur is in host memory (an address below the guest begin) and it returned 0. SMBW asks for this value after recording each material setup handle,
// and on 0 (with control memory also 0) treats the recording as empty and marks the handle invalid (from main+0x4bb6c it clears bit 0 of the material entry +0x50); later draws no longer CALL it and inherit the previous object's vertex buffers / textures:
// on the iPhone a batch of draws on odd frames of the opening course fetched vertices out of bounds, giving huge triangles and NaN (black band, black-edged glow, grey stripes). Without overflow the accumulated value equals cur - begin.
// As in NVN, the count runs from the start of the command memory the game last added (AddCommandMemory), not from BeginRecording: Fire Emblem: Three Houses adds one 16 MB chunk of a ring to its main
// command buffer and records every frame into it; its memory callback (main+0x9ca960) takes chunk start + used as the position the GPU has reached and frees the ring up to there when the frame's fence
// signals. Counted per recording, the position stayed at one frame's size (about 18 KB), the ring never freed past it, and after the first 16 MB (about 15 s) the callback waited for ring space forever
NVN(nvnCommandBufferGetCommandMemoryUsed, size_t, const NVNcommandBuffer *c) { const cmd_t *m = (const cmd_t *)c;
    return m->begin ? (size_t)m->used : 0; }
NVN(nvnCommandBufferGetControlMemoryUsed, size_t, const NVNcommandBuffer *c) { (void)c; return 0; }
NVN(nvnCommandBufferGetControlMemoryFree, size_t, const NVNcommandBuffer *c) { return ((const cmd_t *)c)->ctl_size; }
NVN(nvnCommandBufferGetControlMemorySize, size_t, const NVNcommandBuffer *c) { return ((const cmd_t *)c)->ctl_size; }
// Size and free bytes of the command memory the game last added (DRAGON QUEST XI S, Unreal Engine 4.18): as in NVN, size = used + free; free is what is left in the current segment
NVN(nvnCommandBufferGetCommandMemoryFree, size_t, const NVNcommandBuffer *c) { const cmd_t *m = (const cmd_t *)c; return m->cur && m->end > m->cur ? (size_t)(m->end - m->cur) : 0; }
NVN(nvnCommandBufferGetCommandMemorySize, size_t, const NVNcommandBuffer *c) { return h_nvnCommandBufferGetCommandMemoryUsed(c) + h_nvnCommandBufferGetCommandMemoryFree(c); }
NVN_NOPROTO(nvnCommandBufferSetCommandMemoryCallbackEnabled, void, NVNcommandBuffer *c, uint32_t on) { (void)c; (void)on; }

// nvnProgramSetShadersExt (NVN 55): each stage is 0x18 bytes = NVNshaderData{data, control} + one 8-byte extension field (filled from SMBW main 0x1074ff0).
// The host identifies stages by the control section and does not use the extension field: convert to plain NVNshaderData and hand it to SetShaders.
typedef struct { NVNbufferAddress data; const void *control; uint64_t ext; } shader_data_ext;
NVN_NOPROTO(nvnProgramSetShadersExt, NVNboolean, NVNprogram *p, int n, const shader_data_ext *sd) {
    if (n < 0 || n > 6) { LOG("[RED] ProgramSetShadersExt stage count %d", n); return 0; }
    NVNshaderData plain[6];
    for (int i = 0; i < n; i++) { plain[i].data = sd[i].data; plain[i].control = sd[i].control; if (sd[i].ext) ONCE_LOG("ProgramSetShadersExt extension field nonzero (%#llx): unused by the host", (unsigned long long)sd[i].ext); }
    return h_nvnProgramSetShaders(p, n, plain);
}

// NVN2 (Switch 2) creates programs from a builder: SetDefaults, SetDevice, SetShaders(count, NVNshaderData[]) and then nvnProgramInitializeFromBuilder (Donkey Kong Bananza). The stages are
// 16-byte {data, control} pairs as in NVN, so initializing from the builder is nvnProgramInitialize + nvnProgramSetShaders. The builder only keeps the caller's array, which the game keeps
// until it initializes the program (its stack frame), as other NVN builders keep their pointers. Lichii: the array as the game wrote it (its addresses), which GetShaders returns;
// InitializeFromBuilder turns them into host pointers. sample_shading, dbg, link / nlink: SetSampleShading, SetDebugDataHashes, SetSubroutineLinkage ("Object state and queries");
// rt_*: the ray-tracing setters ("Device command buffers, ... ray-tracing builder state")
typedef struct { NVNdevice *dev; const NVNshaderData *sd; int n, sample_shading, nlink; const void *dbg, *link; int32_t rt_nmod, rt_max_trace, rt_max_callable; uint32_t rt_flags; uint64_t rt_mod; } prog_builder_t;
NVN_NOPROTO(nvnProgramBuilderSetDefaults, void, void *b) { memset(b, 0, sizeof(prog_builder_t)); ((prog_builder_t *)b)->rt_max_trace = ((prog_builder_t *)b)->rt_max_callable = 2; }
NVN_NOPROTO(nvnProgramBuilderSetDevice, void, void *b, NVNdevice *d) { ((prog_builder_t *)b)->dev = d; }
NVN_NOPROTO(nvnProgramBuilderSetShaders, void, void *b, int n, const void *sd) { ((prog_builder_t *)b)->n = n; ((prog_builder_t *)b)->sd = sd; }
NVN_NOPROTO(nvnProgramInitializeFromBuilder, NVNboolean, NVNprogram *p, const void *b) {
    const prog_builder_t *o = (const prog_builder_t *)b;
    if (o->n < 0 || o->n > 6) { LOG("[RED] ProgramInitializeFromBuilder stage count %d", o->n); return 0; }
    NVNshaderData sd[6];
    for (int i = 0; i < o->n; i++) sd[i] = (NVNshaderData){nvn_direct_host_of(o->sd[i].data), (const void *)(uintptr_t)nvn_direct_host_of((uint64_t)(uintptr_t)o->sd[i].control)};
    return h_nvnProgramInitialize(p, o->dev) && (!o->n || h_nvnProgramSetShaders(p, o->n, sd));
}
// A queue NVN2 ties to a primary queue (Donkey Kong Bananza's second queue): the host executes every queue's submits in order on its one Metal queue, so the tie needs no work (kept for the getter)
NVN_NOPROTO(nvnQueueBuilderSetPrimaryQueue, void, NVNqueueBuilder *b, NVNqueue *primary) { QB(b)->primary = primary; }

// ---------- Native images ----------
// A decoded video picture as a texture (nn::gs::NativeImageHandle → nvnNativeImage → nvnTextureInitializeFromNativeImage; Fire Emblem Fortune's Weave's movies, main+0x1900da8, RGBA8). These are not in
// the proc table: the game imports them by name from the sdk. Upstream's handle comes from nn::codec's pool (hle_codec_video.c, not on this branch: dm_native_image is the weak stub below), which keeps the picture as RGBA in the game's pool memory: a linear texture
// over a memory pool of that memory then reads each new picture with no upload. GameChat's camera pictures come the same way (hle_misc.c dm_chat_camera_image: Mario Kart World 1.8.0 makes
// textures of its camera storage at boot and aborts if one fails; with no camera they stay black)
__attribute__((weak)) int dm_chat_camera_image(uint32_t handle, uint8_t **mem, uint64_t *size, uint32_t *w, uint32_t *h, uint32_t *stride) { (void)handle; (void)mem; (void)size; (void)w; (void)h; (void)stride; return 0; }   // hle_misc.c
__attribute__((weak)) int dm_native_image(uint32_t handle, uint8_t **mem, uint64_t *size, uint32_t *w, uint32_t *h, uint32_t *stride) { (void)handle; (void)mem; (void)size; (void)w; (void)h; (void)stride; return 0; }   // hle_codec_video.c; graphics unit tests have no decoder
typedef struct { NVNdevice *dev; uint32_t handle; NVNformat format; } native_builder_t;   // the game gives the builder 0x100 bytes
typedef struct { NVNdevice *dev; uint32_t handle; NVNformat format; uint32_t has_pool; NVNmemoryPool pool; } native_image_t;   // and the image 0x200 (main+0x1900de0: its texture is 0x200 further on)
_Static_assert(sizeof(native_image_t) <= 0x200, "the game's NVNnativeImage");
HLE_FORCE("_Z32nvnNativeImageBuilderSetDefaultsP21NVNnativeImageBuilder", "native images are the host's", void, h_ni_defaults, native_builder_t *b) { NVNdevice *d = b->dev; memset(b, 0, sizeof *b); b->dev = d; }   // the game sets the device first
HLE_FORCE("_Z30nvnNativeImageBuilderSetDeviceP21NVNnativeImageBuilderP9NVNdevice", "native images are the host's", void, h_ni_device, native_builder_t *b, NVNdevice *d) { b->dev = d; }
HLE_FORCE("_Z41nvnNativeImageBuilderSetNativeImageHandleP21NVNnativeImageBuilderN2nn2gs17NativeImageHandleE", "native images are the host's", void, h_ni_handle, native_builder_t *b, uint32_t h) { b->handle = h; }
HLE_FORCE("_Z40nvnNativeImageBuilderSetMemoryPermissionP21NVNnativeImageBuilder30NVNnativeImageMemoryPermission", "native images are the host's: the permission has no meaning there", void, h_ni_perm, native_builder_t *b, int p) { (void)b; (void)p; }
HLE_FORCE("_Z30nvnNativeImageBuilderSetFormatP21NVNnativeImageBuilder9NVNformat", "native images are the host's", void, h_ni_format, native_builder_t *b, NVNformat f) { b->format = f; }
HLE_FORCE("_Z24nvnNativeImageInitializeP14NVNnativeImagePK21NVNnativeImageBuilder", "native images are the host's", NVNboolean, h_ni_init, native_image_t *i, const native_builder_t *b) {
    i->dev = b->dev; i->handle = b->handle; i->format = b->format; i->has_pool = 0;
    return 1;
}
HLE_FORCE("_Z22nvnNativeImageFinalizeP14NVNnativeImage", "native images are the host's", void, h_ni_fini, native_image_t *i) {
    if (i->has_pool) h_nvnMemoryPoolFinalize(&i->pool);
    i->has_pool = 0;
}
HLE_FORCE("_Z35nvnTextureInitializeFromNativeImageP10NVNtextureP14NVNnativeImage", "a linear texture over the picture's memory", NVNboolean, h_ni_texture, NVNtexture *t, native_image_t *i) {
    uint8_t *mem; uint64_t size; uint32_t w, h, stride;
    if (!dm_native_image(i->handle, &mem, &size, &w, &h, &stride) && !dm_chat_camera_image(i->handle, &mem, &size, &w, &h, &stride)) { LOG("[RED] nvnTextureInitializeFromNativeImage: handle %#x names no picture", i->handle); return 0; }
    if (!i->has_pool) {
        // A pool over the game's buffer: placement needs whole pages (an overlap with one of the game's own pools is not checked)
        if (((uintptr_t)mem | size) & 0xFFF) LOG("[RED] nvnTextureInitializeFromNativeImage: handle %#x: picture memory %p of %#llx bytes is not whole 4 KiB pages; its pool may not be placed", i->handle, (void *)mem, (unsigned long long)size);
        NVNmemoryPoolBuilder pb; h_nvnMemoryPoolBuilderSetDefaults(&pb); h_nvnMemoryPoolBuilderSetDevice(&pb, i->dev); h_nvnMemoryPoolBuilderSetStorage(&pb, mem, size);
        h_nvnMemoryPoolBuilderSetFlags(&pb, NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED | NVN_MEMORY_POOL_FLAGS_GPU_CACHED);
        if (!h_nvnMemoryPoolInitialize(&i->pool, &pb)) return 0;
        i->has_pool = 1;
    }
    NVNtextureBuilder tb; h_nvnTextureBuilderSetDefaults(&tb); h_nvnTextureBuilderSetDevice(&tb, i->dev);
    h_nvnTextureBuilderSetTarget(&tb, NVN_TEXTURE_TARGET_2D); h_nvnTextureBuilderSetFormat(&tb, i->format); h_nvnTextureBuilderSetSize2D(&tb, (int)w, (int)h); h_nvnTextureBuilderSetLevels(&tb, 1);
    h_nvnTextureBuilderSetFlags(&tb, NVN_TEXTURE_FLAGS_LINEAR); h_nvnTextureBuilderSetStride(&tb, stride); h_nvnTextureBuilderSetStorage(&tb, &i->pool, 0);
    return h_nvnTextureInitialize(t, &tb);
}

// NVN2 push constants: (command buffer, a word seen 0, byte offset, size, data). The bytes form a constant block of the graphics stages or of compute, which their shaders read
// as c[24] (DMSH_PUSH_BANK; NVN2 compute's own c0 layout starts with them, which the translator maps there). Donkey Kong Bananza's upscaler pushes (1 / w, 1 / h, w, h) of its
// target for its graphics passes, and two integer rectangles (source, destination) for its compute network
static void push_const(NVNcommandBuffer *c, int compute, uint64_t w, uint64_t off, uint64_t n, const void *data) {
    if (w) { static int once; if (!once++) LOG("[RED] push constants with the word before the offset %#llx: its meaning is not known", (unsigned long long)w); }
    if (off + n > DMSH_PUSH_SIZE || !data) { static int k; if (k++ < 8) LOG("[RED] push constants at %#llx + %#llx: beyond the host's %#x-byte block, dropped", (unsigned long long)off, (unsigned long long)n, DMSH_PUSH_SIZE); return; }
    p_push_const *r = RECN(c, OP_PUSH_CONST, p_push_const, data, n); r->compute = (uint16_t)compute; r->offset = (uint16_t)off; r->size = (uint32_t)n; memcpy(r->data, data, n);
}
NVN_NOPROTO(nvnCommandBufferPushGraphicsConstants, void, NVNcommandBuffer *c, uint64_t w, uint64_t off, uint64_t n, const void *data) { push_const(c, 0, w, off, n, data); }
NVN_NOPROTO(nvnCommandBufferPushComputeConstants, void, NVNcommandBuffer *c, uint64_t w, uint64_t off, uint64_t n, const void *data) { push_const(c, 1, w, off, n, data); }
// NVN2 command buffer modes Donkey Kong Bananza sets on every command buffer it begins: they tune how the GPU's front end runs the commands (whether compute may overlap the next
// work, how geometry is processed, how commands are recorded), not what they produce. The host records and replays every command in order on Metal, so none changes its result.
// Logged once each with the first value; open: what the values select on the console
NVN_NOPROTO(nvnCommandBufferSetRecordingMode, void, NVNcommandBuffer *c, int mode) { (void)c; ONCE_LOG("nvnCommandBufferSetRecordingMode %d: the host records every command buffer alike", mode); }
NVN_NOPROTO(nvnCommandBufferSetGeometryProcessingMode, void, NVNcommandBuffer *c, int mode) { (void)c; ONCE_LOG("nvnCommandBufferSetGeometryProcessingMode %d: no effect on Metal", mode); }
NVN_NOPROTO(nvnCommandBufferSetComputeOverlapMode, void, NVNcommandBuffer *c, int mode) { (void)c; ONCE_LOG("nvnCommandBufferSetComputeOverlapMode %d: the host runs compute in submission order", mode); }

// nvnProgramSetSampleShading (NVN 53+): whether the program's fragment shader runs once per sample of a multisampled target. Metroid Prime Remastered sets it on its programs
// (modes 0, 1 and 2, from CGraphicsDeviceContext's sample shading state). Metal has no such switch: a fragment function runs per sample when it reads the sample index ([[sample_id]],
// what PIXLD.MYINDEX translates to) or interpolates at a sample, and once per pixel otherwise, so the mode is not applied; on the single-sample targets met so far the two agree
NVN_NOPROTO(nvnProgramSetSampleShading, NVNboolean, NVNprogram *p, int mode) { (void)p; ONCE_LOG("nvnProgramSetSampleShading mode %d: Metal shades per sample only where the shader reads the sample index", mode); return 1; }

// ---------- Commands used by newer-SDK games (SMBW): most map onto existing record formats, the replay layer is unchanged ----------
// Singular / combined forms → plural / separate forms; state the replay layer does not have yet (conditional render, depth bounds, ...) is logged once and drawing continues (better to over-draw than to drop).
#include "../tex/nvn_deswizzle.h"

NVN(nvnCommandBufferCallCommands, void, NVNcommandBuffer *c, int n, const NVNcommandHandle *h) { h_nvnCommandBufferCopyCommands(c, n, h); }   // the replay layer treats CALL and COPY alike
// Combined texture handle (texture id | sampler id << 20) is recorded into the texture slot as is: the replay layer takes the texture from the low 20 bits and the sampler from the high bits (nvn_mtl.m NVN_RB_SAMPT)
// One combined handle: recorded as the hardware lays the command out, the handle 16 bytes in, where Luigi's Mansion 3 patches it after recording (nvn_obj.h p_bind_tex1)
NVN(nvnCommandBufferBindTexture, void, NVNcommandBuffer *c, NVNshaderStage st, int i, NVNtextureHandle h) {
    int32_t tb = CB(c)->tex_base, sb = CB(c)->samp_base;
    if (tb || sb) { *REC(c, OP_BIND_TEX1, p_bind_tex1) = (p_bind_tex1){st, i, (uint64_t)h, tb, sb}; return; }
    p_bind_tex1 *r = rec(c, OP_BIND_TEX1, offsetof(p_bind_tex1, tbase)); r->stage = st | BIND_TEX1_NOBASE; r->slot = i; r->h = (uint64_t)h;   // no bases: the record ends after h
}
_Static_assert(offsetof(p_bind_tex1, h) == 8, "a single texture bind keeps its handle 16 bytes into the record (8 of header)");
NVN(nvnCommandBufferBindTextures, void, NVNcommandBuffer *c, NVNshaderStage st, int first, int n, const NVNtextureHandle *h) { bind_handles(c, OP_BIND_TEX, st, first, n, (const uint64_t *)h); }
// "Deferred" binding: the GPU reads the handles h points to when it executes this command, not at recording. The per-texture binding of The Legend of Zelda: Breath of the Wild's engine (main+0xec2e64) always uses it,
//   passing the address of the handle field in the texture sampler object with a count of 1; command blocks are recorded once and submitted repeatedly, and the handle is written / replaced afterwards. The host used to copy the handles at recording (mostly 0),
//   so every texture of the deferred lighting composite became the dummy texture 0 and the whole 3D picture was black. The address is recorded and the replay layer reads it when it reads the record (bh_fresh)
NVN_NOPROTO(nvnCommandBufferBindTexturesDeferred, void, NVNcommandBuffer *c, NVNshaderStage st, int first, int n, const NVNtextureHandle *h) {
    p_bind_handles *r = bind_handles(c, OP_BIND_TEX, st, first, n, (const uint64_t *)h);
    if (r && h) { r->defer = 1; r->src = (uint64_t)(uintptr_t)h; }
    if (CB(c)->tex_base || CB(c)->samp_base) ONCE_LOG("[RED] deferred-bound handles are read only at replay: the numbering base of this command buffer's bound descriptor pools is not added (pools %d / %d)", CB(c)->tex_base, CB(c)->samp_base);
}
NVN(nvnCommandBufferBindImage, void, NVNcommandBuffer *c, NVNshaderStage st, int i, NVNimageHandle h) { uint64_t v = (uint64_t)h; bind_handles(c, OP_BIND_IMG, st, i, 1, &v); }
NVN(nvnCommandBufferBindVertexBuffer, void, NVNcommandBuffer *c, int i, NVNbufferAddress a, size_t n) { NVNbufferRange r = {a, n}; h_nvnCommandBufferBindVertexBuffers(c, i, 1, &r); }

// Draws
NVN(nvnCommandBufferDrawElements, void, NVNcommandBuffer *c, NVNdrawPrimitive pr, NVNindexType it, int n, NVNbufferAddress ib) { h_nvnCommandBufferDrawElementsInstanced(c, pr, it, n, ib, 0, 0, 1); }
NVN(nvnCommandBufferDrawElementsBaseVertex, void, NVNcommandBuffer *c, NVNdrawPrimitive pr, NVNindexType it, int n, NVNbufferAddress ib, int bv) { h_nvnCommandBufferDrawElementsInstanced(c, pr, it, n, ib, bv, 0, 1); }
NVN(nvnCommandBufferDrawArraysInstanced, void, NVNcommandBuffer *c, NVNdrawPrimitive pr, int first, int n, int bi, int inst) {
    rec_draw(c, (p_draw){pr, first, n, inst, bi});
}
// Indirect, as DrawArraysIndirect: the record at ind is read by the GPU (nvn_mtl.m draw)
// Indirect-count multi-draws (Monster Hunter Rise's GPU-driven scene: compute culling writes the draw records and a u32 count, then up to max draws are made from them; 1–49 per call,
// stride 20, the records and counts in its sparse virtual pool). Metal has no draw count read by the GPU, so each multi-draw becomes max ordinary indirect draws that carry the count's address
// and their index; the replay masks their records on the GPU, zeros past the count (nvn_mtl.m mdx_run). Both used to be empty, and the whole scene stayed undrawn
enum { MDI_MAX = 1024 };
NVN(nvnCommandBufferMultiDrawElementsIndirectCount, void, NVNcommandBuffer *c, NVNdrawPrimitive pr, NVNindexType it, NVNbufferAddress ib, NVNbufferAddress ind, NVNbufferAddress cnt, int max, ptrdiff_t stride) {
    if (max > MDI_MAX) { ONCE_LOG("[RED] indirect-count multi-draw of up to %d draws: only the first %d are made", max, MDI_MAX); max = MDI_MAX; }
    for (int i = 0; i < max; i++) rec_draw_idx(c, (p_draw_idx){pr, it, 0, 0, 0, 1, ib, ind + (uint64_t)i * (uint64_t)stride, cnt, i, max, (int32_t)stride});
}
NVN(nvnCommandBufferMultiDrawArraysIndirectCount, void, NVNcommandBuffer *c, NVNdrawPrimitive pr, NVNbufferAddress ind, NVNbufferAddress cnt, int max, ptrdiff_t stride) {
    if (max > MDI_MAX) { ONCE_LOG("[RED] indirect-count multi-draw of up to %d draws: only the first %d are made", max, MDI_MAX); max = MDI_MAX; }
    for (int i = 0; i < max; i++) rec_draw(c, (p_draw){pr, 0, 0, 1, 0, ind + (uint64_t)i * (uint64_t)stride, cnt, i, max, (int32_t)stride});
}
// Whether the host can read n bytes at a guest address: outside a sparse virtual pool always; inside one only where pages are mapped (nvn_mtl.m reads multi-draw counts there)
int nvn_host_readable(uint64_t a, uint64_t n) {
    int ok = 1; pthread_mutex_lock(&g_vp_mu);
    for (int k = 0; k < g_nvp; k++) { struct vpool *v = g_vp[k]; if (a < (uint64_t)(uintptr_t)v->base || a >= (uint64_t)(uintptr_t)v->base + v->len) continue;
        uint64_t o = a - (uint64_t)(uintptr_t)v->base; ok = 0;
        for (int r = 0; r < v->nrun; r++) if (o >= v->run[r].lo && o + n <= v->run[r].hi) { ok = 1; break; }
        break; }
    pthread_mutex_unlock(&g_vp_mu); return ok;
}
// Sparse virtual addresses alias physical pools, so guest ranges alone cannot
// prove copy independence.
int nvn_sparse_alias(uint64_t lo, uint64_t hi) {
    if (hi <= lo) return 0;
    int r = 0; pthread_mutex_lock(&g_vp_mu);
    for (int k = 0; k < g_nvp && !r; k++) { const struct vpool *v = g_vp[k];
        uint64_t base = (uint64_t)(uintptr_t)v->base;
        r = lo < base + v->len && base < hi;
    }
    pthread_mutex_unlock(&g_vp_mu); return r;
}
// Textures of a sparse pool, read through its page table (see the sparse pools above): whether a pool is one, a copy of a texture's bytes (unmapped pages read as 0; → bytes not mapped),
// whether its content may have changed since an upload (a mapping in its range newer than gen, or a physical page behind it written after the dirty-page stamp), and the write
// protection of the physical pages behind it (taken before an upload, as for any packaged texture)
// The pool is found by address among the live sparse pools (g_vp, sorted by base), never through the pool object: a texture keeps its pool's pointer, and the game may finalize the
// pool and reuse its memory while the texture is still registered (Mario Kart World on the phone, m18: vka_tex_prepare wrote a stale texture ID of a finalized pool, and its vp was
// the game's new bytes, 0xdb6db6db6db0ff06: SIGSEGV in vpool_at). A finalized pool's range reads as unmapped (zeros), or as the pages of a newer pool reserved at the same address.
// Only the reserved ranges [base, base+len) are disjoint; an address past its pool's reservation (len < size: iOS reserves at most 1 GB) is matched by pool pointer, compared, never read
static struct vpool *vpool_at(const void *pool, uint64_t a) {   // caller holds g_vp_mu
    int lo = 0, hi = g_nvp;
    while (lo < hi) { int m = (lo + hi) / 2; if ((uint64_t)(uintptr_t)g_vp[m]->base <= a) lo = m + 1; else hi = m; }   // the last pool based at or below a
    struct vpool *v = lo ? g_vp[lo - 1] : NULL, *hit = v && a < (uint64_t)(uintptr_t)v->base + v->len && a < (uint64_t)(uintptr_t)v->base + v->size ? v : NULL;
    if (hit && (const void *)hit->pool == pool) return hit;
    for (int k = 0; k < g_nvp; k++) { v = g_vp[k]; if ((const void *)v->pool == pool && a >= (uint64_t)(uintptr_t)v->base && a < (uint64_t)(uintptr_t)v->base + v->size) return v; }
    return hit;   // (a stale pool pointer: the live pool reserved at a, if any)
}
int nvn_sparse_pool(const void *pool) { return pool && ((const pool_t *)pool)->vp && ((const pool_t *)pool)->vmode != 1; }   // no lock: vp set at Initialize, cleared at Finalize, vmode set by MapVirtual; called per texture lookup
uint64_t nvn_sparse_gen(void) { return atomic_load(&g_vp_gen); }
uint64_t nvn_sparse_copy(const void *pool, void *dst, uint64_t a, uint64_t n) {
    pthread_mutex_lock(&g_vp_mu); struct vpool *v = vpool_at(pool, a); uint64_t miss = 0;
    if (!v) { pthread_mutex_unlock(&g_vp_mu); memset(dst, 0, n); return n; }
    for (uint64_t p = 0, o = a - (uint64_t)(uintptr_t)v->base; p < n;) {
        uint64_t at = o + p, k = 0x1000 - (at & 0xFFF); if (k > n - p) k = n - p;
        const vpte *e = at < v->size ? vpt_get(v, at) : NULL;
        const uint8_t *s = e && e->phys ? e->phys + (at & 0xFFF) : NULL;
        // (Lichii: guest memory through its backing memory, which a guest unmap of the physical pages leaves readable; never through nvn_direct_readable's lookup of g_vp)
        if (s && nvn_direct_guest_of((uint64_t)(uintptr_t)s) != (uint64_t)(uintptr_t)s) dm_guest_read_mapped((uint8_t *)dst + p, (uintptr_t)s, k);
        else if (s) memcpy((uint8_t *)dst + p, s, k); else { memset((uint8_t *)dst + p, 0, k); miss += k; }
        p += k;
    }
    pthread_mutex_unlock(&g_vp_mu); return miss;
}
// Runs of consecutive physical pages behind [a, a+n): fn on each (nonzero stops the walk and is returned), or 1 at the first page mapped after gen (gen ~0 = do not look)
static int vpt_walk(const void *pool, uint64_t a, uint64_t n, uint64_t gen, int (*fn)(const uint8_t *, uint64_t, uint64_t), uint64_t arg) {
    pthread_mutex_lock(&g_vp_mu); struct vpool *v = vpool_at(pool, a); int r = 0;
    if (!v) { pthread_mutex_unlock(&g_vp_mu); return 1; }
    const uint8_t *rs = NULL, *re = NULL; uint64_t o = a - (uint64_t)(uintptr_t)v->base;
    for (uint64_t at = o & ~0xFFFull; at < o + n && at < v->size && !r; at += 0x1000) {
        const vpte *e = vpt_get(v, at);
        if (!e) { at |= (2u << 20) - 1; at -= 0xFFF; continue; }   // a leaf never mapped: skip its 2 MB
        if (gen != ~0ull && e->gen > gen) { r = 1; break; }
        if (!e->phys) continue;
        if (rs && e->phys == re) { re += 0x1000; continue; }
        if (rs) r = fn(rs, (uint64_t)(re - rs), arg);
        rs = e->phys; re = rs + 0x1000;
    }
    if (!r && rs) r = fn(rs, (uint64_t)(re - rs), arg);
    pthread_mutex_unlock(&g_vp_mu); return r;
}
static int run_written(const uint8_t *p, uint64_t n, uint64_t stamp) { return nvn_dirty_since(p, n, stamp); }
static int run_protect(const uint8_t *p, uint64_t n, uint64_t unused) { (void)unused; nvn_dirty_protect(p, n); return 0; }
int nvn_sparse_since(const void *pool, uint64_t a, uint64_t n, uint64_t gen, uint64_t stamp) { return vpt_walk(pool, a, n, gen, run_written, stamp); }
int nvn_sparse_mapped(const void *pool, uint64_t a, uint64_t n) {   // every page of [a, a+n) has memory behind it
    pthread_mutex_lock(&g_vp_mu); struct vpool *v = vpool_at(pool, a); int ok = v != NULL;
    for (uint64_t o = v ? (a - (uint64_t)(uintptr_t)v->base) & ~0xFFFull : 0, end = v ? a - (uint64_t)(uintptr_t)v->base + n : 0; ok && o < end; o += 0x1000) { const vpte *e = o < v->size ? vpt_get(v, o) : NULL; ok = e && e->phys; }
    pthread_mutex_unlock(&g_vp_mu); return ok;
}
// Lichii: whether the page at a lies in a virtual pool's range with nothing mapped there (inaccessible to the CPU): nvn_direct_readable answers for those ranges with it
int nvn_vpool_page_unmapped(uint64_t a) {
    pthread_mutex_lock(&g_vp_mu);
    int lo = 0, hi = g_nvp; while (lo < hi) { int m = (lo + hi) / 2; if ((uint64_t)(uintptr_t)g_vp[m]->base <= a) lo = m + 1; else hi = m; }
    const struct vpool *v = lo ? g_vp[lo - 1] : NULL; int r = 0;
    if (v && a < (uint64_t)(uintptr_t)v->base + v->len) { uint64_t o = a - (uint64_t)(uintptr_t)v->base; const vpte *e = o < v->size ? vpt_get(v, o) : NULL; r = !e || !e->phys; }
    pthread_mutex_unlock(&g_vp_mu); return r;
}
void nvn_sparse_protect(const void *pool, uint64_t a, uint64_t n) { vpt_walk(pool, a, n, ~0ull, run_protect, 0); }
static int run_any(const uint8_t *p, uint64_t n, uint64_t unused) { (void)p; (void)n; (void)unused; return 1; }
int nvn_sparse_any(const void *pool, uint64_t a, uint64_t n) {   // some page of [a, a+n) has memory behind it (leaves never mapped are skipped 2 MB at a time)
    pthread_mutex_lock(&g_vp_mu); int in = vpool_at(pool, a) != NULL; pthread_mutex_unlock(&g_vp_mu);
    return in && vpt_walk(pool, a, n, ~0ull, run_any, 0);
}
NVN(nvnCommandBufferDrawElementsIndirect, void, NVNcommandBuffer *c, NVNdrawPrimitive pr, NVNindexType it, NVNbufferAddress ib, NVNbufferAddress ind) {
    rec_draw_idx(c, (p_draw_idx){pr, it, 0, 0, 0, 1, ib, ind});
}

// Viewport / scissor / depth range: single and multiple forms
NVN(nvnCommandBufferSetViewport, void, NVNcommandBuffer *c, int x, int y, int w, int h) { float v[4] = {(float)x, (float)y, (float)w, (float)h}; h_nvnCommandBufferSetViewports(c, 0, 1, v); }
NVN(nvnCommandBufferSetScissors, void, NVNcommandBuffer *c, int first, int n, const int *r) {
    if (first == 0 && n >= 1) h_nvnCommandBufferSetScissor(c, r[0], r[1], r[2], r[3]);
    if (first + n > 1) ONCE_LOG("SetScissors first=%d n=%d: the host uses only index 0", first, n);
}
NVN(nvnCommandBufferSetDepthRanges, void, NVNcommandBuffer *c, int first, int n, const float *r) {
    if (first == 0 && n >= 1) h_nvnCommandBufferSetDepthRange(c, r[0], r[1]);
    if (first + n > 1) ONCE_LOG("SetDepthRanges first=%d n=%d: the host uses only index 0", first, n);
}

// GPU value write: ReportValue = write a 32-bit value to the address when this point is reached (the replay layer's event write, applied on completion)
NVN(nvnCommandBufferReportValue, void, NVNcommandBuffer *c, uint32_t v, NVNbufferAddress a) {
    uint8_t *p = cmd_alloc(CB(c), 24); put_hdr(p, OP_SIGNAL_EVENT, 24);
    volatile uint32_t *val = (volatile uint32_t *)(uintptr_t)a; uint32_t mm = NVN_EVENT_SIGNAL_MODE_WRITE;
    memcpy(p + 8, &val, 8); memcpy(p + 16, &mm, 4); memcpy(p + 20, &v, 4);
}
// Clear a buffer to a 32-bit value: one OP_CLEAR_BUF record, filled in order within the submission at replay (fillBuffer when all bytes match, otherwise copied from a pattern buffer)
NVN(nvnCommandBufferClearBuffer, void, NVNcommandBuffer *c, NVNbufferAddress a, size_t n, uint32_t v) { *REC(c, OP_CLEAR_BUF, p_clear_buf) = (p_clear_buf){a, n, v, 0}; }

// State the replay layer does not have yet: logged once (conditional rendering is treated as "always draw")
// Render enable (profile render_cond, see OP_RENDER_COND in nvn_mtl.m): recorded and evaluated at replay; without the key everything draws, as before
NVN(nvnCommandBufferSetRenderEnable, void, NVNcommandBuffer *c, uint32_t en) {
    if (DM_GAME_RENDER_COND) *REC(c, OP_RENDER_COND, p_render_cond) = (p_render_cond){(en & 0xff) ? 2 : 3, 0, 0};
    else if (!(en & 0xff)) ONCE_LOG("[RED] SetRenderEnable(false): the host draws anyway for now");
}
NVN(nvnCommandBufferSetRenderEnableConditional, void, NVNcommandBuffer *c, NVNconditionalRenderMode m, NVNbufferAddress a) {
    if (DM_GAME_RENDER_COND) *REC(c, OP_RENDER_COND, p_render_cond) = (p_render_cond){(int)m & 1, 0, a};
    else ONCE_LOG("[RED] SetRenderEnableConditional: the host draws anyway for now");
}
// Fire Emblem: Three Houses turns depth clamp on for its scenes; ignored, the ground right under its close-up battle cameras was clipped away and the lighting pass painted the sky's fog colour there
NVN(nvnCommandBufferSetDepthClamp, void, NVNcommandBuffer *c, uint32_t en) { REC(c, OP_DEPTH_CLAMP, p_depth_clamp)->on = (en & 0xff) != 0; }
// Absolum (FNA3D's NVN driver) keeps the lower-left window origin and sets viewport 0's swizzle every frame (NEGATIVE_Y: clip y negated, how a GL-heritage renderer flips its picture).
// Ignored, its whole picture came out upside down. The replay applies it to the viewport (draw in nvn_mtl.m)
NVN(nvnCommandBufferSetViewportSwizzles, void, NVNcommandBuffer *c, int first, int n, const NVNviewportSwizzle *s) {
    if (n < 0 || first < 0 || first + n > 16) { LOG("[RED] SetViewportSwizzles first=%d n=%d", first, n); return; }
    p_vp_swizzle *r = REC(c, OP_VP_SWIZZLE, p_vp_swizzle); r->first = first; r->n = n;
    for (int i = 0; i < n; i++) for (int k = 0; k < 4; k++) r->sw[i][k] = (uint8_t)((s[4 * i + k] ^ 2 * k) & 7);
}
NVN(nvnCommandBufferSetAlphaRef, void, NVNcommandBuffer *c, float r) { REC(c, OP_ALPHA_REF, p_alpha_ref)->ref = r; }
NVN(nvnCommandBufferSetLineWidth, void, NVNcommandBuffer *c, float w) { (void)c; if (w != 1.0f) ONCE_LOG("SetLineWidth %.2f: Metal has only 1-pixel lines", w); }
NVN(nvnCommandBufferSetPointSize, void, NVNcommandBuffer *c, float s) { (void)c; (void)s; ONCE_LOG("SetPointSize: the point size comes from the shader"); }
NVN(nvnCommandBufferSetPrimitiveRestart, void, NVNcommandBuffer *c, uint32_t en, int idx) { *REC(c, OP_PRIM_RESTART, p_prim_restart) = (p_prim_restart){(en & 0xff) != 0, (uint32_t)idx}; }
// Rasterizer discard: draws run their vertex stage only (with transform feedback, the capture); recorded, and at replay such a draw rasterizes nothing (nvn_mtl.m)
NVN(nvnCommandBufferSetRasterizerDiscard, void, NVNcommandBuffer *c, uint32_t en) { REC(c, OP_RAST_DISCARD, p_rast_discard)->on = (en & 0xff) != 0; }
NVN(nvnCommandBufferSetTiledCacheAction, void, NVNcommandBuffer *c, NVNtiledCacheAction a) { (void)c; (void)a; }        // Tegra tiled cache hint, no Metal equivalent
// A bound target's contents may be discarded: they are undefined afterwards, so keeping them is correct (The Great Ace Attorney Chronicles discards colour target 0)
NVN(nvnCommandBufferDiscardColor, void, NVNcommandBuffer *c, int i) { (void)c; (void)i; }
NVN(nvnCommandBufferDiscardDepthStencil, void, NVNcommandBuffer *c) { (void)c; }
NVN(nvnCommandBufferSetTiledCacheTileSize, void, NVNcommandBuffer *c, int w, int h) { (void)c; (void)w; (void)h; }
NVN_NOPROTO(nvnCommandBufferSetColorReductionEnable, void, NVNcommandBuffer *c, uint32_t en) { (void)c; (void)en; }   // color compression hints
NVN_NOPROTO(nvnCommandBufferSetColorReductionThresholds, void, NVNcommandBuffer *c, int cls, const int *t) { (void)c; (void)cls; (void)t; }
// Resolve a multisample texture into a single-sample one (whole texture): recorded as a texture copy with DM_COPY_RESOLVE; replay uses Metal's multisample resolve (copy_t2t in nvn_mtl.m)
static void rec_downsample(NVNcommandBuffer *c, const NVNtexture *s, const NVNtextureView *sv, const NVNtexture *d, const NVNtextureView *dv, uint64_t ra) {
    p_copy_t2t *o = REC_SNAP(c, OP_COPY_T2T, p_copy_t2t, 2);
    o->src = tex_snap(o, SNAP_OFF(p_copy_t2t), 0, s, ra); o->dst = tex_snap(o, SNAP_OFF(p_copy_t2t), 1, d, ra);
    copy_view(&o->sv, &o->has_sv, sv); copy_view(&o->dv, &o->has_dv, dv);
    if (o->src) o->sr = (NVNcopyRegion){0, 0, 0, (int)o->src->d.width, (int)o->src->d.height, 1};
    if (o->dst) o->dr = (NVNcopyRegion){0, 0, 0, (int)o->dst->d.width, (int)o->dst->d.height, 1};
    o->flags = DM_COPY_RESOLVE;
}
NVN(nvnCommandBufferDownsample, void, NVNcommandBuffer *c, const NVNtexture *s, const NVNtexture *d) { rec_downsample(c, s, NULL, d, NULL, nvn_caller()); }
NVN(nvnCommandBufferDownsampleTextureView, void, NVNcommandBuffer *c, const NVNtexture *s, const NVNtextureView *sv, const NVNtexture *d, const NVNtextureView *dv) {
    rec_downsample(c, s, sv, d, dv, nvn_caller());
}

// CPU texture write (NVN WriteTexels): texture storage lives in pool memory in Tegra block-linear layout; decode the whole level → patch the region → re-encode.
// Write protection is lifted and the range marked dirty first, so the replay layer re-uploads it the next time it is used
// Non-3D textures: only the bytes inside the region are written, by block-linear address, without decoding the whole level. Metroid Dread's minimap (main+0x160e80) calls WriteTexels once per texel (a 1×1 region);
// previously every call decoded, patched and re-encoded the whole level, and after a few minutes the main thread spent whole frames there (frames stopped from 175 s). Returns −1 when the region is out of range or the format unsupported, falling back to the whole-level path
static int write_region_bl(const tex_t *t, unsigned level, unsigned layer, const NVNcopyRegion *r, const uint8_t *src, size_t row, int bw, int bh, int bpb) {
    uint32_t row_bytes, rows, slices, gbpb; uint64_t base; unsigned th, td;
    if (nvn_deswizzle_geom(&t->d, level, layer, &row_bytes, &rows, &slices, &base, &th, &td, &gbpb) || (int)gbpb != bpb || r->xoffset < 0 || r->yoffset < 0) return -1;
    uint32_t x0 = (uint32_t)(r->xoffset / bw) * (uint32_t)bpb, y0 = (uint32_t)(r->yoffset / bh);
    size_t wb = (size_t)((r->width + bw - 1) / bw) * (size_t)bpb, hb = (size_t)((r->height + bh - 1) / bh);
    if (x0 + wb > row_bytes || y0 + hb > rows) return -1;
    uint32_t lb = nvn_tex_level_gobs(&t->d, level) * 64; if (lb < row_bytes) lb = row_bytes;   // the row as laid out (padded in a sparse texture's tiled levels)
    for (size_t y = 0; y < hb; y++)
        for (size_t x = 0; x < wb;) {   // 16-byte runs are contiguous within a GOB
            uint32_t bx = x0 + (uint32_t)x; size_t n = 16 - bx % 16; if (n > wb - x) n = wb - x;
            memcpy(t->mem + base + nvn_bl_offset(bx, y0 + (uint32_t)y, 0, lb, rows, th, td), src + y * row + x, n); x += n;
        }
    return 0;
}
static void write_texels(const tex_t *t, const view_t *v, const NVNcopyRegion *r, const uint8_t *src, size_t row, size_t img) {
    if (!t->mem) { LOG("[RED] WriteTexels: texture has no storage"); return; }
    int bw, bh, bpb = nvn_fmt_bpb(t->d.format, &bw, &bh);
    if (!bpb) { ONCE_LOG("[RED] WriteTexels: unknown format %#x", t->d.format); return; }
    unsigned level = v && v->set_levels ? (unsigned)v->base_level : 0, layer0 = v && v->set_layers ? (unsigned)v->min_layer : 0;
    int is3d = t->d.target == NVN_TEXTURE_TARGET_3D, nz = r->depth ? r->depth : 1;
    if (!is3d) {
        size_t wb0 = (size_t)((r->width + bw - 1) / bw) * bpb, row0 = row ? row : wb0, img0 = img ? img : row0 * (size_t)((r->height + bh - 1) / bh);
        dm_guest_write_prepare(t->mem, t->L.size);
        int ok = 1;
        for (int z = 0; z < nz && ok; z++) ok = !write_region_bl(t, level, layer0 + (unsigned)(r->zoffset + z), r, src + (size_t)z * img0, row0, bw, bh, bpb);
        if (ok) { mtl_tex_cpu_written(t); return; }
    }
    uint32_t lw = t->d.width >> level ? t->d.width >> level : 1, lh = t->d.height >> level ? t->d.height >> level : 1, lz = is3d ? (t->d.depth >> level ? t->d.depth >> level : 1) : 1;
    size_t lp = (size_t)((lw + bw - 1) / bw) * bpb, lr = (lh + bh - 1) / bh;
    size_t wb = (size_t)((r->width + bw - 1) / bw) * bpb, hb = (size_t)(r->height + bh - 1) / bh;
    if (!row) row = wb;
    if (!img) img = row * hb;
    uint8_t *lin = malloc(lp * lr * lz); if (!lin) die("WriteTexels allocation failed");
    dm_guest_write_prepare(t->mem, t->L.size);
    // when the region is exactly the whole level (2D) every texel is overwritten, so the old content need not be decoded first (movies write whole frames; saves one deswizzle per 1080p frame)
    int whole = !is3d && !r->xoffset && !r->yoffset && (uint32_t)r->width == lw && (uint32_t)r->height == lh;
    for (int z = 0; z < nz; z++) {
        unsigned layer = is3d ? 0 : layer0 + (unsigned)(r->zoffset + z);
        if (((is3d && z == 0) || !is3d) && !whole) if (nvn_deswizzle(&t->d, level, layer, t->mem, lin, lp)) { ONCE_LOG("[RED] WriteTexels deswizzle failed (format %#x target %d)", t->d.format, t->d.target); free(lin); return; }
        uint8_t *dst = lin + (is3d ? (size_t)(r->zoffset + z) * lp * lr : 0) + (size_t)(r->yoffset / bh) * lp + (size_t)(r->xoffset / bw) * bpb;
        for (size_t y = 0; y < hb; y++) memcpy(dst + y * lp, src + (size_t)z * img + y * row, wb);
        if (!is3d || z == nz - 1) nvn_swizzle(&t->d, level, layer, lin, lp, t->mem);
    }
    free(lin);
    mtl_tex_cpu_written(t);
}
NVN(nvnTextureWriteTexels, void, const NVNtexture *t, const NVNtextureView *v, const NVNcopyRegion *r, const void *src) { write_texels(CTX(t), (const view_t *)v, r, src, 0, 0); }
// with the source's row and image strides in bytes (0 = tightly packed, as WriteTexels)
NVN(nvnTextureWriteTexelsStrided, void, const NVNtexture *t, const NVNtextureView *v, const NVNcopyRegion *r, const void *src, ptrdiff_t row, ptrdiff_t img) {
    write_texels(CTX(t), (const view_t *)v, r, src, (size_t)row, (size_t)img);
}
// GPU copies into a texture the host uploads from its storage (tex_mtl_'s packaged textures: a packaged layout, CPU-written, or registered by tex_flush_upload / tex_dirty_upload and not
// resolved yet): the hardware's copy writes the storage, so the host writes the copied bytes there as well (copy_b2t / copy_t2t in nvn_mtl.m), and an upload from the storage later
// carries them. Yoshi's Crafted World's texture streamer re-creates a texture with more levels, copies the resident levels from the old texture on the GPU (Unreal's
// RHIAsyncReallocateTexture2D: nvnCommandBufferCopyTextureToTexture, a level at a time) and writes the new, finer levels from the CPU; uploaded from the storage, the copied levels held
// what the pool had there before (another texture's data, zeros), and the Yoshis' felt sampled speckles from the moment a level streamed in. −1 when the copy cannot be mirrored (3D,
// a format mismatch, a region out of range): the GPU copy alone then stands until the next upload
static int mirror_prepare(const tex_t *t, unsigned level, unsigned layer) {   // unprotect (and mark written) the storage of one level of one layer
    uint64_t lo = nvn_tex_level_offset(&t->d, level), hi = level + 1 < t->d.levels ? nvn_tex_level_offset(&t->d, level + 1) : t->L.layer_size;
    if (hi <= lo || hi > t->L.layer_size || (uint64_t)layer * t->L.layer_size + hi > t->L.size) return -1;
    dm_guest_write_prepare(t->mem + (uint64_t)layer * t->L.layer_size + lo, hi - lo);
    return 0;
}
int nvn_tex_mirror_b2t(const tex_t *t, unsigned level, unsigned layer0, const NVNcopyRegion *r, const uint8_t *src, size_t row, size_t img) {
    int bw, bh, bpb = nvn_fmt_bpb(t->d.format, &bw, &bh);
    if (!t->mem || !bpb || t->d.target == NVN_TEXTURE_TARGET_3D || level >= t->d.levels) return -1;
    int nz = r->depth ? r->depth : 1;
    for (int z = 0; z < nz; z++) {
        unsigned layer = layer0 + (unsigned)(r->zoffset + z);
        if (mirror_prepare(t, level, layer) || write_region_bl(t, level, layer, r, src + (size_t)z * img, row, bw, bh, bpb)) return -1;
    }
    return 0;
}
int nvn_tex_mirror_t2t(const tex_t *s, unsigned sl, unsigned ss, const NVNcopyRegion *sr, const tex_t *d, unsigned dl, unsigned ds, const NVNcopyRegion *dr) {
    int bw, bh, bpb = nvn_fmt_bpb(s->d.format, &bw, &bh), dbw, dbh, dbpb = nvn_fmt_bpb(d->d.format, &dbw, &dbh);
    if (!s->mem || !d->mem || !bpb || bpb != dbpb || bw != dbw || bh != dbh || s->d.target == NVN_TEXTURE_TARGET_3D || d->d.target == NVN_TEXTURE_TARGET_3D) return -1;
    if (sl >= s->d.levels || dl >= d->d.levels || sr->width != dr->width || sr->height != dr->height || sr->xoffset < 0 || sr->yoffset < 0) return -1;
    uint32_t lw = s->d.width >> sl ? s->d.width >> sl : 1, lh = s->d.height >> sl ? s->d.height >> sl : 1;
    size_t lp = (size_t)((lw + bw - 1) / bw) * bpb, lr = (lh + bh - 1) / bh, x0 = (size_t)(sr->xoffset / bw) * bpb, y0 = (size_t)(sr->yoffset / bh);
    if (x0 + (size_t)((sr->width + bw - 1) / bw) * bpb > lp || y0 + (size_t)((sr->height + bh - 1) / bh) > lr) return -1;
    uint8_t *lin = malloc(lp * lr); if (!lin) die("texture copy mirror allocation failed");
    int nz = sr->depth ? sr->depth : 1, ok = 1;
    for (int z = 0; z < nz && ok; z++) {
        unsigned layer = ds + (unsigned)(dr->zoffset + z);
        ok = !nvn_deswizzle(&s->d, sl, ss + (unsigned)(sr->zoffset + z), s->mem, lin, lp) && !mirror_prepare(d, dl, layer) && !write_region_bl(d, dl, layer, dr, lin + y0 * lp + x0, lp, bw, bh, bpb);
    }
    free(lin);
    return ok ? 0 : -1;
}
// The movie decoder's native-texture output (DecoderMode 2, hle_movie.c): a whole frame into level 0 of the client texture, as WriteTexels does; returns the texture format, or −1 when the object is not a texture
int nvn_host_write_texture(const void *nvntex, const void *src, int w, int h, size_t row, int *tw, int *th) {
    const tex_t *t = nvntex;
    uint32_t mg = 0;   // the value the client gives is not necessarily a pointer: test-read first
    if (!t || (uintptr_t)t & 7 || plat_mem_read((uint64_t)(uintptr_t)t, &mg, 4)) return -1;
    if (t->magic != MAGIC('T', 'E', 'X', 0)) return -1;
    if (tw) *tw = (int)t->d.width;
    if (th) *th = (int)t->d.height;
    if (src) { NVNcopyRegion r = {0, 0, 0, w, h, 1}; write_texels(t, NULL, &r, src, row, 0); }
    return (int)t->d.format;
}
// work the host finished on the CPU: the sync object counts as signalled at once (fence 0 is always complete)
int nvn_host_signal_sync(void *nvnsync) {
    sync_t *o = nvnsync;
    if (!o || o->magic != MAGIC('S', 'Y', 'N', 'C')) return -1;
    o->fence = 0; o->fenced = 1; return 0;
}
NVN(nvnTextureFlushTexels, void, const NVNtexture *t, const NVNtextureView *v, const NVNcopyRegion *r) { (void)t; (void)v; (void)r; }   // unified memory + dirty page tracking: nothing to flush

// ---------- Object state and queries ----------
// Getters return what the matching setter stored, or the driver's default, and the queries answer as NVN2's driver does (Mario Kart World's and Ocarina of Time's sdk, read
// statically). Where the driver keeps state the host has no use for (debug labels, checkpoints, GL interop), it answers as the driver does without them.

// Device
NVN_NOPROTO(nvnDeviceBuilderGetFlags, int, const NVNdeviceBuilder *b) { return (int)((const dev_builder_t *)b)->flags; }
NVN_NOPROTO(nvnDeviceGetFlags, int, const NVNdevice *d) { return (int)((const dev_t_ *)d)->flags; }
NVN(nvnDeviceSetDebugLabel, void, NVNdevice *d, const char *l) { (void)d; (void)l; }   // (the driver keeps no label either)
// The driver's IDs count up from 0, process-wide
NVN(nvnDeviceGenerateDebugDomainId, NVNdebugDomainId, const NVNdevice *d, const char *name) { static atomic_int next; (void)d; (void)name; return atomic_fetch_add(&next, 1); }
NVN(nvnDeviceWalkDebugDatabase, void, const NVNdevice *d, NVNdebugObjectType t, PFNNVNWALKDEBUGDATABASECALLBACKPROC cb, void *data) { (void)d; (void)t; (void)cb; (void)data; }   // the release driver's is empty
NVN(nvnDeviceIsExternalDebuggerAttached, NVNboolean, const NVNdevice *d) { (void)d; return 0; }
// NVN1's device with deferred finalizes (NVN_DEVICE_FLAG_DEFERRED_FINALIZE): the host finalizes every object at once, so none waits here
NVN(nvnDeviceApplyDeferredFinalizes, void, NVNdevice *d, int age) { (void)d; (void)age; }
// Checkpoints (GPU progress markers for crash dumps): the release driver answers true and keeps nothing; it ignores its arguments, the device the only one read here
NVN_NOPROTO(nvnDeviceSetAutomaticCheckpointsEnable, NVNboolean, NVNdevice *d) { (void)d; return 1; }
NVN_NOPROTO(nvnDeviceSetAutomaticCheckpointsSamplingInterval, NVNboolean, NVNdevice *d) { (void)d; return 1; }
NVN_NOPROTO(nvnDeviceSetAutomaticCheckpointsByDebugGroup, NVNboolean, NVNdevice *d) { (void)d; return 1; }
NVN_NOPROTO(nvnDeviceSetAutomaticCheckpointsMask, NVNboolean, NVNdevice *d) { (void)d; return 1; }
NVN_NOPROTO(nvnDeviceSetCheckpointCallStacksEnable, NVNboolean, NVNdevice *d) { (void)d; return 1; }
NVN_NOPROTO(nvnDeviceSetCheckpointMaxCallStacksDepth, NVNboolean, NVNdevice *d) { (void)d; return 1; }
// Fast clear values: on the console a clear to a registered value is a compression hint, which the host does not need. The driver keeps a table per device and answers each
// registration with the value's index in it (−1 when it fails); the host keeps its own table (dev_t_ fc_*), with room for NVN_FAST_CLEARS values of each kind: a value
// registered again keeps its index. A depth index is only given to 0 and 1 (Ocarina of Time's driver); colours are matched by their four words, whatever the format
static int fast_clear(uint32_t *table, int32_t *n, int words, const uint32_t *v) {
    for (int i = 0; i < *n; i++) if (!memcmp(table + i * words, v, (size_t)words * 4)) return i;
    if (*n == NVN_FAST_CLEARS) return -1;
    memcpy(table + *n * words, v, (size_t)words * 4); return (*n)++;
}
static int fast_clear_color(NVNdevice *d, const void *c) { dev_t_ *o = (dev_t_ *)d; return fast_clear(&o->fc_color[0][0], &o->fc_ncolor, 4, c); }
static int fast_clear_depth(NVNdevice *d, float v) { dev_t_ *o = (dev_t_ *)d; uint32_t u; memcpy(&u, &v, 4); return v == 0.f || v == 1.f ? fast_clear(o->fc_depth, &o->fc_ndepth, 1, &u) : -1; }
static int fast_clear_stencil(NVNdevice *d, int s) { dev_t_ *o = (dev_t_ *)d; uint32_t u = (uint32_t)s & 0xff; return fast_clear(o->fc_stencil, &o->fc_nstencil, 1, &u); }
NVN(nvnDeviceRegisterFastClearColori, NVNboolean, NVNdevice *d, const int *c, NVNformat f) { (void)f; return fast_clear_color(d, c) >= 0; }
NVN(nvnDeviceRegisterFastClearColorui, NVNboolean, NVNdevice *d, const uint32_t *c, NVNformat f) { (void)f; return fast_clear_color(d, c) >= 0; }
NVN_NOPROTO(nvnDeviceRegisterFastClearColorIndex, int, NVNdevice *d, const float *c, NVNformat f) { (void)f; return fast_clear_color(d, c); }
NVN_NOPROTO(nvnDeviceRegisterFastClearColoriIndex, int, NVNdevice *d, const int *c, NVNformat f) { (void)f; return fast_clear_color(d, c); }
NVN_NOPROTO(nvnDeviceRegisterFastClearColoruiIndex, int, NVNdevice *d, const uint32_t *c, NVNformat f) { (void)f; return fast_clear_color(d, c); }
NVN_NOPROTO(nvnDeviceRegisterFastClearDepthIndex, int, NVNdevice *d, float v) { return fast_clear_depth(d, v); }
NVN_NOPROTO(nvnDeviceRegisterFastClearStencil, NVNboolean, NVNdevice *d, int s) { return fast_clear_stencil(d, s) >= 0; }
NVN_NOPROTO(nvnDeviceRegisterFastClearStencilIndex, int, NVNdevice *d, int s, NVNformat f) { (void)f; return fast_clear_stencil(d, s); }
// The bytes of command memory a recording takes: its records up to its END, through the JUMPs that link its segments (the driver's handle carries this size)
NVN_NOPROTO(nvnDeviceGetCommandHandleMemorySize, size_t, const NVNdevice *d, NVNcommandHandle h) {
    (void)d;
    size_t n = 0;
    for (const uint8_t *p = (const uint8_t *)(uintptr_t)h; p;) {
        op_hdr r; memcpy(&r, p, sizeof r);
        if (r.size < 8 || r.size & 7) { LOG("[RED] DeviceGetCommandHandleMemorySize: handle %#llx has a record of %u bytes", (unsigned long long)h, r.size); return n; }
        n += r.size;
        if (r.op == OP_END) break;
        if (r.op == OP_JUMP) { memcpy(&p, p + 8, 8); continue; }
        p += r.size;
    }
    return n;
}
// A bindless uniform buffer handle as NVN2's driver encodes it: the buffer's size in 16-byte units from bit 45, its address in 16-byte units below (the game's address)
NVN_NOPROTO(nvnDeviceGetUniformBufferHandle, uint64_t, const NVNdevice *d, NVNbufferAddress a, size_t size) {
    (void)d; return (((uint64_t)size + 15) >> 4 << 45) | nvn_direct_guest_of(a) >> 4;
}

// Queue. NVN's counters measure the queue's own memory (SetQueueMemory): the bytes the driver wrote there for the queue's submissions since the last reset (command and control
// rings; compute: its own count). The host never writes the queue's memory (its submissions replay the command buffers' records), so the counts are 0 and a reset has nothing to note
NVN(nvnQueueGetTotalCommandMemoryUsed, size_t, NVNqueue *q) { (void)q; return 0; }
NVN(nvnQueueGetTotalControlMemoryUsed, size_t, NVNqueue *q) { (void)q; return 0; }
NVN(nvnQueueGetTotalComputeMemoryUsed, size_t, NVNqueue *q) { (void)q; return 0; }
NVN(nvnQueueResetMemoryUsageCounts, void, NVNqueue *q) { (void)q; }
#define CQB(b) ((const queue_builder_t *)(b))
NVN_NOPROTO(nvnQueueBuilderGetDevice, NVNdevice *, const NVNqueueBuilder *b) { return CQB(b)->dev; }
NVN_NOPROTO(nvnQueueBuilderGetFlags, int, const NVNqueueBuilder *b) { return (int)CQB(b)->flags; }
NVN_NOPROTO(nvnQueueBuilderGetQueuePriority, int, const NVNqueueBuilder *b) { return CQB(b)->prio; }
NVN_NOPROTO(nvnQueueBuilderGetPrimaryQueue, NVNqueue *, const NVNqueueBuilder *b) { return CQB(b)->primary; }
// The size as set (0 unset) and whether it was set
static NVNboolean qb_get(const NVNqueueBuilder *b, int k, int *size) { *size = (int)CQB(b)->sz[k]; return CQB(b)->set[k]; }
NVN_NOPROTO(nvnQueueBuilderGetCommandMemorySize, NVNboolean, const NVNqueueBuilder *b, int *size) { return qb_get(b, QB_CMD, size); }
NVN_NOPROTO(nvnQueueBuilderGetComputeMemorySize, NVNboolean, const NVNqueueBuilder *b, int *size) { return qb_get(b, QB_COMP, size); }
NVN_NOPROTO(nvnQueueBuilderGetControlMemorySize, NVNboolean, const NVNqueueBuilder *b, int *size) { return qb_get(b, QB_CTL, size); }
NVN_NOPROTO(nvnQueueBuilderGetCommandFlushThreshold, NVNboolean, const NVNqueueBuilder *b, int *size) { return qb_get(b, QB_FLUSH, size); }
NVN_NOPROTO(nvnQueueBuilderGetMemorySize, int, const NVNqueueBuilder *b) { return (int)CQB(b)->mem_sz; }
NVN_NOPROTO(nvnQueueBuilderGetMemory, void *, const NVNqueueBuilder *b) { return CQB(b)->mem; }

// Memory pool, buffer, texture and sampler builders and objects. Debug IDs: the release driver's are 0
#define CPB(b) ((const pool_builder_t *)(b))
NVN_NOPROTO(nvnMemoryPoolBuilderGetDevice, NVNdevice *, const NVNmemoryPoolBuilder *b) { return CPB(b)->dev; }
NVN_NOPROTO(nvnMemoryPoolBuilderGetMemory, void *, const NVNmemoryPoolBuilder *b) { return CPB(b)->mem; }   // (nvn_protos.h has it returning void)
NVN(nvnMemoryPoolBuilderGetSize, size_t, const NVNmemoryPoolBuilder *b) { return (size_t)CPB(b)->size; }
NVN(nvnMemoryPoolBuilderGetFlags, NVNmemoryPoolFlags, const NVNmemoryPoolBuilder *b) { return (NVNmemoryPoolFlags)CPB(b)->flags; }
NVN_NOPROTO(nvnBufferBuilderGetDevice, NVNdevice *, const NVNbufferBuilder *b) { return ((const buf_builder_t *)b)->dev; }
NVN(nvnBufferGetDebugID, uint64_t, const NVNbuffer *buf) { (void)buf; return 0; }
NVN(nvnTextureGetDebugID, uint64_t, const NVNtexture *t) { (void)t; return 0; }
NVN(nvnSamplerGetDebugID, uint64_t, const NVNsampler *s) { (void)s; return 0; }
#define CTB(b) ((const tex_builder_t *)(b))
NVN_NOPROTO(nvnTextureBuilderGetDevice, NVNdevice *, const NVNtextureBuilder *b) { return CTB(b)->dev; }
NVN(nvnTextureBuilderGetSwizzle, void, const NVNtextureBuilder *b, NVNtextureSwizzle *r, NVNtextureSwizzle *g, NVNtextureSwizzle *bl, NVNtextureSwizzle *a) {
    unsigned w = CTB(b)->set_swz ? CTB(b)->swz : 2u | 3u << 3 | 4u << 6 | 5u << 9;   // identity R G B A unless set (tex_swz)
    *r = (NVNtextureSwizzle)(w & 7); *g = (NVNtextureSwizzle)(w >> 3 & 7); *bl = (NVNtextureSwizzle)(w >> 6 & 7); *a = (NVNtextureSwizzle)(w >> 9 & 7);
}
NVN(nvnTextureBuilderGetDepthStencilMode, NVNtextureDepthStencilMode, const NVNtextureBuilder *b) { return CTB(b)->stencil ? NVN_TEXTURE_DEPTH_STENCIL_MODE_STENCIL : NVN_TEXTURE_DEPTH_STENCIL_MODE_DEPTH; }
NVN(nvnTextureBuilderGetPackagedTextureData, const void *, const NVNtextureBuilder *b) { return CTB(b)->pkg_data; }
// The driver keeps the layout's first 12 bits (the block height and the two fields after it) and writes them back as one 32-bit word; returns whether a layout was set
NVN_NOPROTO(nvnTextureBuilderGetPackagedTextureLayout, NVNboolean, const NVNtextureBuilder *b, NVNpackagedTextureLayout *l) {
    const tex_builder_t *o = CTB(b);
    uint32_t w = o->has_layout ? (uint32_t)(o->layout[0] | o->layout[1] << 8) & 0xfff : 0;
    memcpy(l->layout, &w, 4);
    return o->has_layout;
}
NVN(nvnTextureBuilderGetGLTextureName, uint32_t, const NVNtextureBuilder *b) { return CTB(b)->gl_name; }
// The view's offset into the storage the builder describes, as nvnTextureGetViewOffset gives it for a texture
NVN_NOPROTO(nvnTextureBuilderGetViewOffset, ptrdiff_t, const NVNtextureBuilder *b, const NVNtextureView *v) {
    const view_t *w = (const view_t *)v;
    if (!w) return 0;
    nvn_texlayout L; nvn_tex_layout(&CTB(b)->d, &L);
    uint64_t off = w->set_levels ? nvn_tex_level_offset(&CTB(b)->d, (unsigned)w->base_level) : 0;
    if (w->set_layers) off += (uint64_t)w->min_layer * L.layer_size;
    return (ptrdiff_t)off;
}
#define CSB(b) (&((const samp_builder_t *)(b))->p)
NVN_NOPROTO(nvnSamplerBuilderGetDevice, NVNdevice *, const NVNsamplerBuilder *b) { return ((const samp_builder_t *)b)->dev; }
NVN(nvnSamplerBuilderGetLodBias, float, const NVNsamplerBuilder *b) { return CSB(b)->lod_bias; }
NVN(nvnSamplerBuilderGetCompare, void, const NVNsamplerBuilder *b, NVNcompareMode *m, NVNcompareFunc *f) { *m = (NVNcompareMode)CSB(b)->cmp_mode; *f = (NVNcompareFunc)CSB(b)->cmp_func; }
NVN(nvnSamplerBuilderGetMaxAnisotropy, float, const NVNsamplerBuilder *b) { return CSB(b)->aniso; }
NVN(nvnSamplerBuilderGetReductionFilter, NVNsamplerReduction, const NVNsamplerBuilder *b) { return (NVNsamplerReduction)CSB(b)->reduction; }
NVN(nvnSamplerBuilderGetLodSnap, float, const NVNsamplerBuilder *b) { return ((const samp_builder_t *)b)->lod_snap; }
// The border colour's four words as set, whichever setter set them (the driver keeps the words alone)
NVN(nvnSamplerBuilderGetBorderColor, void, const NVNsamplerBuilder *b, float *c) { memcpy(c, CSB(b)->border, 16); }
NVN(nvnSamplerBuilderGetBorderColori, void, const NVNsamplerBuilder *b, int *c) { memcpy(c, CSB(b)->border, 16); }
NVN(nvnSamplerBuilderGetBorderColorui, void, const NVNsamplerBuilder *b, uint32_t *c) { memcpy(c, CSB(b)->border, 16); }
NVN(nvnSamplerGetBorderColori, void, const NVNsampler *s, int *c) { memcpy(c, SP(s)->border, 16); }
NVN(nvnSamplerGetBorderColorui, void, const NVNsampler *s, uint32_t *c) { memcpy(c, SP(s)->border, 16); }

// State objects
NVN(nvnBlendStateSetAdvancedNormalizedDst, void, NVNblendState *s, uint32_t n) { ST(blend_st, s)->adv_dst_free = !B(n); }   // kept: the replay's fixed-function factors read a normalized target as it is
NVN(nvnBlendStateGetAdvancedNormalizedDst, NVNboolean, const NVNblendState *s) { return !((const blend_st *)s)->adv_dst_free; }
NVN_NOPROTO(nvnMultisampleStateSetCoverageTruncationEnable, void, NVNmultisampleState *s, uint32_t e) { ST(ms_st, s)->cov_trunc = B(e) != 0; }   // kept: Vulkan has no coverage truncation
NVN_NOPROTO(nvnMultisampleStateGetCoverageTruncationEnable, NVNboolean, const NVNmultisampleState *s) { return ((const ms_st *)s)->cov_trunc; }
NVN_NOPROTO(nvnMultisampleStateGetSampleLocations, void, const NVNmultisampleState *s, int start, int n, float *v) {
    for (int i = 0; i < n; i++) {
        uint8_t l = (unsigned)(start + i) < 16 ? ((const ms_loc_st *)s)->loc[start + i] : 0x88;
        v[2 * i] = (float)(l & 15) / 16.f; v[2 * i + 1] = (float)(l >> 4) / 16.f;
    }
}
// The pixel grid the sample locations repeat over, by the larger of the sample and raster sample counts; other counts write nothing (the driver's table)
NVN(nvnMultisampleStateGetSampleLocationsGrid, void, NVNmultisampleState *s, int *w, int *h) {
    const ms_st *o = (const ms_st *)s; unsigned n = o->samples > o->raster_samples ? o->samples : o->raster_samples;
    int gw, gh;
    switch (n) { case 0: case 1: gw = 4; gh = 4; break; case 2: gw = 2; gh = 4; break; case 4: gw = 2; gh = 2; break; case 8: gw = 1; gh = 2; break; case 16: gw = 1; gh = 1; break; default: return; }
    *w = gw; *h = gh;
}

// Command buffer
NVN(nvnCommandBufferGetCopyRowStride, ptrdiff_t, const NVNcommandBuffer *c) { return (ptrdiff_t)((const cmd_t *)c)->copy_row; }
NVN(nvnCommandBufferGetCopyImageStride, ptrdiff_t, const NVNcommandBuffer *c) { return (ptrdiff_t)((const cmd_t *)c)->copy_image; }
NVN(nvnCommandBufferGetMemoryCallback, PFNNVNCOMMANDBUFFERMEMORYCALLBACKPROC, const NVNcommandBuffer *c) { return ((const cmd_t *)c)->cb; }
NVN_NOPROTO(nvnCommandBufferGetMemoryCallbackData, void *, const NVNcommandBuffer *c) { return ((const cmd_t *)c)->cb_data; }   // (nvn_protos.h has it returning void)

// Syncs and GL: there is no GL on the host. Initializing from a GL sync fails as the driver's does when it has no GL: the sync is initialized unfenced and the error is 4;
// a GL sync of an NVN sync fails with 4, the driver's answer for a sync it cannot express (and, fenced, after writing 0 for the GL sync)
NVN_NOPROTO(nvnSyncInitializeFromFencedGLSync, NVNboolean, NVNsync *s, NVNdevice *d, uint32_t gl, int *err) {
    (void)gl; h_nvnSyncInitialize(s, d);
    if (err) *err = 4;
    return 0;
}
NVN_NOPROTO(nvnSyncCreateGLSync, int, const NVNsync *s, uint64_t *gl) {
    if (((const sync_t *)s)->fenced) *gl = 0;
    return 4;
}
NVN_NOPROTO(nvnEventBuilderGetStorage, NVNmemoryPool *, const NVNeventBuilder *b, int64_t *off) { *off = ((const ev_builder_t *)b)->off; return (NVNmemoryPool *)((const ev_builder_t *)b)->pool; }
NVN_NOPROTO(nvnEventGetMemoryPool, NVNmemoryPool *, const NVNevent *e) { return (NVNmemoryPool *)((const ev_t *)e)->pool; }
NVN_NOPROTO(nvnEventGetMemoryOffset, int64_t, const NVNevent *e) { return ((const ev_t *)e)->off; }

// Program builder (prog_builder_t): the arrays are the game's, returned as it gave them
#define CPRB(b) ((const prog_builder_t *)(b))
NVN_NOPROTO(nvnProgramBuilderGetDevice, NVNdevice *, const void *b) { return CPRB(b)->dev; }
NVN_NOPROTO(nvnProgramBuilderGetShaders, const NVNshaderData *, const void *b) { return CPRB(b)->sd; }
NVN_NOPROTO(nvnProgramBuilderGetShaderCount, int, const void *b) { return CPRB(b)->n; }
NVN_NOPROTO(nvnProgramBuilderSetSampleShading, void, void *b, int mode) { ((prog_builder_t *)b)->sample_shading = mode; }   // kept: see nvnProgramSetSampleShading
NVN_NOPROTO(nvnProgramBuilderGetSampleShading, int, const void *b) { return CPRB(b)->sample_shading; }
NVN_NOPROTO(nvnProgramBuilderSetDebugDataHashes, void, void *b, int n, const void *hashes) { (void)n; ((prog_builder_t *)b)->dbg = hashes; }   // (no getter reads the count)
NVN_NOPROTO(nvnProgramBuilderGetDebugDataHashes, const void *, const void *b) { return CPRB(b)->dbg; }
NVN_NOPROTO(nvnProgramBuilderSetSubroutineLinkage, void, void *b, int n, const void *maps) { ((prog_builder_t *)b)->nlink = n; ((prog_builder_t *)b)->link = maps; }
NVN_NOPROTO(nvnProgramBuilderGetSubroutineLinkage, const void *, const void *b) { return CPRB(b)->link; }
NVN_NOPROTO(nvnProgramBuilderGetSubroutineLinkageCount, int, const void *b) { return CPRB(b)->nlink; }
// A ray tracing program's scratch per warp (the driver asks its ray tracing pipeline, made from the builder's ray tracing modules, for its stack size): 0 for any other program,
// and the host makes no ray tracing programs
NVN_NOPROTO(nvnProgramGetScratchBytesPerWarp, int, const NVNprogram *p) { (void)p; return 0; }

// ---------- Device command buffers, device message buffers, ray-tracing builder state ----------
#include <stdarg.h>
// None of these calls is in nvn.h or Banana Native's prototypes. Signatures, object layouts and defaults are derived from the NVN2 driver in
// Ocarina of Time's sdk NSO (function offsets in that NSO below); the objects keep the driver's layout so a game reading one finds what it wrote.

// nvnDeviceCommandBuffer (sdk+0x657870..0x6578d0): {device, pool, offset, size, command memory size, compute memory size}. GetHandle is the storage's
// GPU address (pool + offset), which NVN2 hands to GPU work that writes commands there.
typedef struct { NVNdevice *dev; const pool_t *pool; int64_t off; uint64_t size, cmd_sz, comp_sz; } dev_cmd_t;
NVN_NOPROTO(nvnDeviceCommandBufferSetDefaults, void, void *b) { memset((uint8_t *)b + 8, 0, sizeof(dev_cmd_t) - 8); }   // the device stays, as in the driver
NVN_NOPROTO(nvnDeviceCommandBufferSetDevice, void, void *b, NVNdevice *d) { ((dev_cmd_t *)b)->dev = d; }
NVN_NOPROTO(nvnDeviceCommandBufferSetStorage, void, void *b, const NVNmemoryPool *p, int64_t off, uint64_t size) {
    dev_cmd_t *o = b; o->pool = (const pool_t *)p; o->off = off; o->size = size;
}
NVN_NOPROTO(nvnDeviceCommandBufferSetCommandMemorySize, void, void *b, uint64_t n) { ((dev_cmd_t *)b)->cmd_sz = n; }
NVN_NOPROTO(nvnDeviceCommandBufferSetComputeMemorySize, void, void *b, uint64_t n) { ((dev_cmd_t *)b)->comp_sz = n; }
NVN_NOPROTO(nvnDeviceCommandBufferGetHandle, NVNcommandHandle, const void *b) {
    const dev_cmd_t *o = b;
    return o->pool ? (NVNcommandHandle)(uintptr_t)(o->pool->mem + o->off) : 0;
}

// nvnDeviceMessageBuffer (sdk+0x6580f0..0x6581e0): shader messages (printf-style) in a buffer of a pool. The buffer starts with a 12-byte header
// {next write offset, size, end at a wrap (0: none)}; each message is {u32 size, u16 kind, s16 code, u32 data[4], format string, arguments}.
// Reading goes through the pool's CPU mapping, as the driver does.
typedef struct { NVNdevice *dev; const pool_t *pool; int32_t off, size; uint8_t *base; uint32_t read, end; uint64_t reserved; } dev_msg_buf_t;
typedef struct { int32_t code, kind; uint32_t data[4]; } dev_msg_t;   // NVNdeviceMessage as GetNextMessage fills it (0x18 bytes)
static uint8_t *dev_msg_mem(const dev_msg_buf_t *o) {
    const pool_t *p = o->pool;
    return p && p->mem && !(p->flags & NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS) ? p->mem + o->off : NULL;
}
NVN_NOPROTO(nvnDeviceMessageBufferSetDefaults, void, void *b) { memset((uint8_t *)b + 8, 0, sizeof(dev_msg_buf_t) - 8); }
NVN_NOPROTO(nvnDeviceMessageBufferSetDevice, void, void *b, NVNdevice *d) { ((dev_msg_buf_t *)b)->dev = d; }
NVN_NOPROTO(nvnDeviceMessageBufferSetStorage, void, void *b, const NVNmemoryPool *p, int off, int size) {
    dev_msg_buf_t *o = b; o->pool = (const pool_t *)p; o->off = off; o->size = size;
}
NVN_NOPROTO(nvnDeviceMessageBufferReset, void, void *b) {
    dev_msg_buf_t *o = b; uint8_t *m = dev_msg_mem(o);
    if (m) { const uint32_t hdr[3] = {12, (uint32_t)o->size, 0}; memcpy(m, hdr, sizeof hdr); }
}
NVN_NOPROTO(nvnDeviceMessageBufferBeginMessages, void, void *b) {
    dev_msg_buf_t *o = b; o->base = dev_msg_mem(o); o->read = 12; o->end = 0;
    if (o->base) { uint32_t hdr[3]; memcpy(hdr, o->base, sizeof hdr); o->end = hdr[2] ? hdr[2] : hdr[0]; }
}
NVN_NOPROTO(nvnDeviceMessageBufferEndMessages, void, void *b) { dev_msg_buf_t *o = b; o->base = NULL; o->read = o->end = 0; }
// The driver's message kinds (u16) as NVN's
static int32_t dev_msg_kind(uint32_t k) {
    switch (k) {
    case 0: return 0; case 1: return 3; case 2: return 4; case 3: return 2; case 4: return 1; case 5: return 5;
    case 0x1493: case 0x14f4: return 6; case 0x1494: case 0x14f5: return 7;
    default: return k >= 0x14c1 && k <= 0x14c6 ? 8 : -1;
    }
}
typedef struct { char *p; size_t left; } dev_msg_out;   // left includes the terminating NUL's byte
static void dev_msg_put(dev_msg_out *w, const char *s, size_t n) {
    if (w->left <= 1) return;
    if (n > w->left - 1) n = w->left - 1;
    memcpy(w->p, s, n); w->p += n; w->left -= n;
}
static void dev_msg_putf(dev_msg_out *w, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static void dev_msg_putf(dev_msg_out *w, const char *fmt, ...) {
    if (w->left <= 1) return;
    va_list ap; va_start(ap, fmt); int n = vsnprintf(w->p, w->left, fmt, ap); va_end(ap);
    if (n < 0) return;
    size_t k = (size_t)n < w->left - 1 ? (size_t)n : w->left - 1; w->p += k; w->left -= k;
}
// The driver's formatter (sdk+0x6583e0..0x658b30): text is copied; %% is a %; a conversion is '%', digits, optionally '.' and digits, optionally
// v2 / v3 / v4 (a vector: that many arguments, ", " between them), then a float conversion (a A e E f F g G: 32-bit floats), d i o u x X (32-bit),
// l followed by d i o u x X or u followed by l (64-bit). Anything else drops the '%' and what was read of the conversion. Arguments follow the
// string from the next 4-byte boundary up to the end of the message; past it they read as 0.
static void dev_msg_format(dev_msg_out *w, const char *s, const uint8_t *arg, const uint8_t *arg_end) {
    while (*s) {
        const char *lit = s; while (*s && *s != '%') s++;
        dev_msg_put(w, lit, (size_t)(s - lit));
        if (!*s) break;
        if (s[1] == '%') { dev_msg_put(w, "%", 1); s += 2; continue; }
        const char *spec = s++;
        while (*s >= '0' && *s <= '9') s++;
        if (*s == '.') { s++; while (*s >= '0' && *s <= '9') s++; }
        char f[40]; size_t fl = (size_t)(s - spec) < 32 ? (size_t)(s - spec) : 32; memcpy(f, spec, fl);
        int n = 1;
        if (*s == 'v') { if (s[1] >= '2' && s[1] <= '4') { n = s[1] - '0'; s += 2; } else { s++; continue; } }
        int wide = 0, flt = 0; char c = *s;
        if (c && strchr("aAeEfFgG", c)) { flt = 1; f[fl++] = c; s++; }
        else if (c && strchr("diouxX", c)) {
            if (c == 'u' && s[1] == 'l') { wide = 1; memcpy(f + fl, "llu", 3); fl += 3; s += 2; } else { f[fl++] = c; s++; }
        } else if (c == 'l' && s[1] && strchr("diouxX", s[1])) { wide = 1; f[fl++] = 'l'; f[fl++] = 'l'; f[fl++] = s[1]; s += 2; }
        else continue;   // resumes as text at c
        f[fl] = 0;
        for (int i = 0; i < n; i++) {
            if (i) dev_msg_put(w, ", ", 2);
            size_t sz = wide ? 8 : 4; uint64_t v = 0;
            if (arg + sz <= arg_end) { memcpy(&v, arg, sz); arg += sz; } else arg = arg_end;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wformat-nonliteral"
            if (flt) { float x; uint32_t u = (uint32_t)v; memcpy(&x, &u, 4); dev_msg_putf(w, f, (double)x); }
            else if (wide) dev_msg_putf(w, f, (unsigned long long)v);
            else dev_msg_putf(w, f, (uint32_t)v);
#pragma clang diagnostic pop
        }
    }
}
NVN_NOPROTO(nvnDeviceMessageBufferGetNextMessage, NVNboolean, void *b, void *msg, char *buf, uint64_t bufsize) {
    dev_msg_buf_t *o = b;
    if (o->read >= o->end) return 0;
    uint8_t *m = dev_msg_mem(o);
    if (!m) return 0;
    const uint8_t *r = m + o->read;
    uint32_t size, data[4]; uint16_t kind; int16_t code;
    memcpy(&size, r, 4); memcpy(&kind, r + 4, 2); memcpy(&code, r + 6, 2); memcpy(data, r + 8, 16);
    dev_msg_t out = {code, dev_msg_kind(kind), {data[0], data[1], data[2], data[3]}};
    memcpy(msg, &out, sizeof out);
    const char *fmt = (const char *)r + 24;
    dev_msg_out w = {buf, bufsize};
    dev_msg_format(&w, fmt, (const uint8_t *)fmt + (strlen(fmt) & ~(size_t)3) + 4, r + size);
    if (bufsize) *w.p = 0;
    o->read += size;
    return 1;
}
// CommandBufferResetDeviceMessageBuffer (sdk+0x6591e0): the GPU writes the empty header in order with the commands around it, an inline write as
// UpdateUniformBuffer records
NVN_NOPROTO(nvnCommandBufferResetDeviceMessageBuffer, void, NVNcommandBuffer *c, const void *b) {
    const dev_msg_buf_t *o = b;
    const uint32_t hdr[3] = {12, (uint32_t)o->size, 0};
    if (o->pool) h_nvnCommandBufferUpdateUniformBuffer(c, (NVNbufferAddress)(uintptr_t)(o->pool->mem + o->off), sizeof hdr, 0, sizeof hdr, hdr);
}

// Ray-tracing state of the program builder (sdk+0x661d60 SetDefaults: both recursion depths 2; +0x661e50..0x661e70, +0x673b90..0x673be0). The
// module array is kept as the game's address, as the driver keeps the pointer.
#define PB(b) ((prog_builder_t *)(b))
NVN_NOPROTO(nvnProgramBuilderSetRayTracingModules, void, void *b, int n, uint64_t modules) { PB(b)->rt_nmod = n; PB(b)->rt_mod = modules; }
NVN_NOPROTO(nvnProgramBuilderGetRayTracingModuleCount, int, const void *b) { return PB(b)->rt_nmod; }
NVN_NOPROTO(nvnProgramBuilderGetRayTracingModuleData, uint64_t, const void *b) { return PB(b)->rt_mod; }
NVN_NOPROTO(nvnProgramBuilderSetMaxTraceRecursionDepth, void, void *b, int n) { PB(b)->rt_max_trace = n; }
NVN_NOPROTO(nvnProgramBuilderGetMaxTraceRecursionDepth, int, const void *b) { return PB(b)->rt_max_trace; }
NVN_NOPROTO(nvnProgramBuilderSetMaxCallableRecursionDepth, void, void *b, int n) { PB(b)->rt_max_callable = n; }
NVN_NOPROTO(nvnProgramBuilderGetMaxCallableRecursionDepth, int, const void *b) { return PB(b)->rt_max_callable; }
NVN_NOPROTO(nvnProgramBuilderSetProgramRayTracingFlags, void, void *b, uint32_t f) { PB(b)->rt_flags = f; }
NVN_NOPROTO(nvnProgramBuilderGetProgramRayTracingFlags, uint32_t, const void *b) { return PB(b)->rt_flags; }
#undef PB

// Queue ray-tracing memory (sdk+0x664500, +0x664820, +0x667760): Get answers whether Set was called and the size set. The queue memory size does not
// include it (sdk+0x664740). The replay runs no ray tracing, so a queue has used none of it.
NVN_NOPROTO(nvnQueueBuilderSetRayTracingMemorySize, void, NVNqueueBuilder *b, int n) { QB(b)->sz[QB_RT] = n; QB(b)->set[QB_RT] = 1; }
NVN_NOPROTO(nvnQueueBuilderGetRayTracingMemorySize, NVNboolean, const NVNqueueBuilder *b, int *n) { *n = ((const queue_builder_t *)b)->sz[QB_RT]; return ((const queue_builder_t *)b)->set[QB_RT]; }
NVN_NOPROTO(nvnQueueGetTotalRayTracingMemoryUsed, uint64_t, const NVNqueue *q) { (void)q; return 0; }

// ---------- DLSS, dirty tile maps and cooperative-vector matrices ----------
// Lichii. nvn.h has no prototypes for these: the signatures, the object layouts and every rule below were read from the NVN2 driver in Ocarina of Time's sdk NSO
// (OoT's driver has all of them; Mario Kart World's lacks the dirty tile map parameters and the cooperative vectors), at the sdk offsets named. The objects are
// kept in the game's memory in the driver's layout, so their getters read back what the setters stored.

// The DLSS network builder (nvnDlssNetworkBuilderSetDefaults sdk+0x65ab90 and its setters): 0xc0 bytes
typedef struct {
    NVNdevice *dev;                  // 0x00 (SetDefaults keeps it)
    const NVNtexturePool *pool;      // 0x08 SetTextureIDs
    void *mem; uint64_t mem_size;    // 0x10 SetStorage
    uint32_t flags; int32_t method, preset, w, h, max_entries;   // 0x20; w, h: the output resolution; max_entries 1 by default
    int32_t ids[32];                 // 0x38 SetTextureIDs: the texture pool entries the driver's network uses
    uint64_t debug;                  // 0xb8 zeroed by SetDefaults (Initialize reads a debug block through it)
} dlss_builder_t;
_Static_assert(sizeof(dlss_builder_t) == 0xc0 && offsetof(dlss_builder_t, ids) == 0x38, "the driver's DLSS network builder");
// The network: the driver's takes 0x258 bytes (nvnDlssNetworkInitialize sdk+0x659b40 clears that much). The host keeps what it was built with, its output resolution
// (Resize) and, for the replay and the log, a serial, its evaluations and history resets
typedef struct { dlss_builder_t b; uint64_t serial, evaluations, resets; int32_t out_w, out_h; } dlss_net_t;
enum { DLSS_NET_BYTES = 0x258 };
_Static_assert(sizeof(dlss_net_t) <= DLSS_NET_BYTES, "the DLSS network exceeds the driver's");
// The evaluation parameters (nvnDlssParametersSetDefaults sdk+0x65ad60 and its setters): 0xc0 bytes
typedef struct {
    const NVNtexture *color; const NVNtextureView *color_view;         // 0x00
    const NVNtexture *output; const NVNtextureView *output_view;       // 0x10
    const NVNtexture *motion; const NVNtextureView *motion_view;       // 0x20
    const NVNtexture *depth; const NVNtextureView *depth_view;         // 0x30
    const NVNtexture *bias; const NVNtextureView *bias_view;           // 0x40 the bias current colour mask
    float jitter[2], mv_scale[2], pre_exposure; uint32_t _pad0;        // 0x50
    const NVNtexture *exposure; const NVNtextureView *exposure_view;   // 0x68
    uint8_t reset, _pad1[3]; int32_t eval_index;                       // 0x78
    int32_t in_size[2], color_base[2], depth_base[2], mv_base[2], bias_base[2], out_base[2];   // 0x80 subrectangles
    uint32_t overlay_debug_layer, overlay_full_screen, overlay_show_nans, overlay_jitter_debug;   // 0xb0, all ~0 by default
} dlss_params_t;
_Static_assert(sizeof(dlss_params_t) == 0xc0 && offsetof(dlss_params_t, jitter) == 0x50 && offsetof(dlss_params_t, exposure) == 0x68 && offsetof(dlss_params_t, reset) == 0x78 &&
               offsetof(dlss_params_t, in_size) == 0x80 && offsetof(dlss_params_t, out_base) == 0xa8 && offsetof(dlss_params_t, overlay_debug_layer) == 0xb0, "the driver's DLSS parameters");
// The network's flags are NGX's DLSS feature flags (0x1 HDR, 0x40 auto exposure; evaluations take the pre-exposure only without either, sdk+0x65a980) plus the driver's
// own: bit 28 makes evaluations take the output subrectangle's base (sdk+0x65a9d0), bit 29 adds storage (GetStorageSize)
enum { DLSS_FLAG_OUTPUT_BASE = 1u << 28, DLSS_FLAG_EXTRA_STORAGE = 1u << 29 };

NVN_NOPROTO(nvnDlssNetworkBuilderSetDefaults, void, void *b) { dlss_builder_t *o = b; NVNdevice *d = o->dev; memset(o, 0, sizeof *o); o->dev = d; o->max_entries = 1; }
NVN_NOPROTO(nvnDlssNetworkBuilderSetDevice, void, void *b, NVNdevice *d) { ((dlss_builder_t *)b)->dev = d; }
NVN_NOPROTO(nvnDlssNetworkBuilderGetDevice, NVNdevice *, const void *b) { return ((const dlss_builder_t *)b)->dev; }
NVN_NOPROTO(nvnDlssNetworkBuilderSetFlags, void, void *b, uint32_t f) { ((dlss_builder_t *)b)->flags = f; }
NVN_NOPROTO(nvnDlssNetworkBuilderGetFlags, uint32_t, const void *b) { return ((const dlss_builder_t *)b)->flags; }
NVN_NOPROTO(nvnDlssNetworkBuilderSetOutputResolution, void, void *b, int w, int h) { ((dlss_builder_t *)b)->w = w; ((dlss_builder_t *)b)->h = h; }
NVN_NOPROTO(nvnDlssNetworkBuilderGetOutputResolution, void, const void *b, int *w, int *h) { if (w) *w = ((const dlss_builder_t *)b)->w; if (h) *h = ((const dlss_builder_t *)b)->h; }
NVN_NOPROTO(nvnDlssNetworkBuilderSetMethod, void, void *b, int m) { ((dlss_builder_t *)b)->method = m; }
NVN_NOPROTO(nvnDlssNetworkBuilderGetMethod, int, const void *b) { return ((const dlss_builder_t *)b)->method; }
NVN_NOPROTO(nvnDlssNetworkBuilderSetPreset, void, void *b, int p) { ((dlss_builder_t *)b)->preset = p; }
NVN_NOPROTO(nvnDlssNetworkBuilderGetPreset, int, const void *b) { return ((const dlss_builder_t *)b)->preset; }
NVN_NOPROTO(nvnDlssNetworkBuilderSetMaxEvaluationEntries, void, void *b, int n) { ((dlss_builder_t *)b)->max_entries = n; }
NVN_NOPROTO(nvnDlssNetworkBuilderGetMaxEvaluationEntries, int, const void *b) { return ((const dlss_builder_t *)b)->max_entries; }
NVN_NOPROTO(nvnDlssNetworkBuilderSetStorage, void, void *b, void *mem, size_t n) { ((dlss_builder_t *)b)->mem = mem; ((dlss_builder_t *)b)->mem_size = n; }
NVN_NOPROTO(nvnDlssNetworkBuilderGetMemory, void *, const void *b) { return ((const dlss_builder_t *)b)->mem; }
NVN_NOPROTO(nvnDlssNetworkBuilderGetMemorySize, size_t, const void *b) { return ((const dlss_builder_t *)b)->mem_size; }
// The pool and its 32 entries (sdk+0x65acb0: 0x80 bytes of IDs). The driver writes the descriptors of its network's textures there; the host's network has none
NVN_NOPROTO(nvnDlssNetworkBuilderSetTextureIDs, void, void *b, const NVNtexturePool *pool, const int *ids) { dlss_builder_t *o = b; o->pool = pool; if (ids) memcpy(o->ids, ids, sizeof o->ids); }
NVN_NOPROTO(nvnDlssNetworkBuilderGetTextureIDs, void, const void *b, int *ids) { if (ids) memcpy(ids, ((const dlss_builder_t *)b)->ids, sizeof ((const dlss_builder_t *)b)->ids); }
NVN_NOPROTO(nvnDlssNetworkBuilderGetTexturePool, const NVNtexturePool *, const void *b) { return ((const dlss_builder_t *)b)->pool; }
// The output is this many times the input by the method (Fire Emblem Fortune's Weave's choice at main+0x1913c24: 0 for 2x, 1 for 3x, 2 for 1x; from nxnative's hle_nvn.c, MIT)
static int dlss_factor(int method) { return method == 1 ? 3 : method == 2 ? 1 : 2; }
static int dlss_ceil_div(int v, int d) { return (v + d - 1) / d; }
// sdk+0x659ae0 asks the driver's DLSS backend with the builder's output size and method, and passes six pointers: the optimal input size, then the smallest and the
// largest input (Xenoblade Chronicles 2 checks lower <= input <= upper before it evaluates: nxnative). The backend's numbers are NVIDIA's; the host answers the method's
// scale and the range of DLSS's modes, a third of the output (ultra performance) up to the output (DLAA), which its upscale takes alike
NVN_NOPROTO(nvnDlssNetworkBuilderGetOptimalSettings, NVNboolean, const void *b, int *ow, int *oh, int *minw, int *minh, int *maxw, int *maxh) {
    const dlss_builder_t *o = b; const int f = dlss_factor(o->method);
    if (ow) *ow = dlss_ceil_div(o->w, f);
    if (oh) *oh = dlss_ceil_div(o->h, f);
    if (minw) *minw = dlss_ceil_div(o->w, 3);
    if (minh) *minh = dlss_ceil_div(o->h, 3);
    if (maxw) *maxw = o->w;
    if (maxh) *maxh = o->h;
    return 1;
}
static uint64_t dlss_texture_bytes(const dlss_builder_t *b, NVNformat f) {   // page-aligned, as the driver sizes its network's textures
    NVNtextureBuilder tb; h_nvnTextureBuilderSetDefaults(&tb); h_nvnTextureBuilderSetDevice(&tb, b->dev);
    h_nvnTextureBuilderSetSize2D(&tb, b->w, b->h); h_nvnTextureBuilderSetFormat(&tb, f);
    return ((uint64_t)h_nvnTextureBuilderGetStorageSize(&tb) + 0xfff) & ~0xfffull;
}
// sdk+0x659fc0, as the driver computes it, so the game sets aside what the console's would. Its inputs: the method (1 or not), the preset (8 or 9 or not), the output
// size, flag bit 29, the maximum evaluation entries (64 KB each), the storage of two output-sized textures (RG16F, RGBA16F) and two bits of the device's internal
// state (device+8 & 0xc00) the host does not have: taken as 0, which on the console selects the smaller sizes
static uint64_t dlss_storage_size(const dlss_builder_t *b) {
    #define PAGE_DOWN(v) ((v) & ~0xfffull)
    const uint64_t w = (uint64_t)(int64_t)b->w, h = (uint64_t)(int64_t)b->h, entries = (uint64_t)(int64_t)b->max_entries << 16;
    const int m1 = b->method == 1, p8 = (b->preset & ~1) == 8, extra = (b->flags & DLSS_FLAG_EXTRA_STORAGE) != 0;
    const uint64_t s = (w + 0x40) * (h + 0x40), wide = (2 * w + 0x80) * (h + 0x40);
    uint64_t x, y, z, r;
    if (p8) x = m1 ? (entries | 0x2000) : 2 * PAGE_DOWN(wide + 0x357ff) + entries + 0x4000;
    else x = dlss_texture_bytes(b, NVN_FORMAT_RG16F) + entries + 0x3000;
    y = m1 ? 0xdc000 + x + 0x1b000 : p8 ? 0xdc000 + x + 0x1b000 : 0xef000 + x + 0x8000;
    if (extra) y += PAGE_DOWN(s + 0x63abf) + PAGE_DOWN(s + 0xfff) + 0x2000;
    y += 0x27000;
    if (extra) y += PAGE_DOWN(s + 0xfff) + 0x1000;
    z = m1 ? 0xdc000 + y + 0x2000 : p8 ? 0xdc000 + y + PAGE_DOWN(s + 0xfff) + 0x3000 : 0xdc000 + y + 0x2000;
    if (extra) z += PAGE_DOWN(wide + 0xc657f) + 0x1000;
    if (!p8) z += dlss_texture_bytes(b, NVN_FORMAT_RGBA16F) + 0x1000;
    const uint64_t t = PAGE_DOWN(s + 0x2c82d), q = z + t + 0x1000, u = PAGE_DOWN(7 * s + 0x7b6fff);
    r = (!m1 && p8 ? 2 * PAGE_DOWN(2 * s + 0x357ff) + q + 0x2000 : q) + u;
    if (p8) r += t + 0xc000;
    else r = dlss_texture_bytes(b, NVN_FORMAT_RG16F) + t + r + 0x5000 + 0x8000 + PAGE_DOWN(7 * s + 0x7f53ff) + 0x1000;
    return (r + 0xfff) & ~0xfffull;
    #undef PAGE_DOWN
}
NVN_NOPROTO(nvnDlssNetworkBuilderGetStorageSize, size_t, const void *b) { return (size_t)dlss_storage_size(b); }
// sdk+0x659b40 clears the network and the storage it is given (it makes a memory pool of it for its network): done alike. The backend's failure is its only false; the
// host's stand-in does not fail
NVN_NOPROTO(nvnDlssNetworkInitialize, NVNboolean, void *n, const void *b) {
    dlss_net_t *o = n; const dlss_builder_t *bb = b;
    memset(o, 0, DLSS_NET_BYTES);
    o->b = *bb; o->serial = nvn_new_serial(); o->out_w = bb->w; o->out_h = bb->h;
    if (bb->mem && bb->mem_size) { dm_guest_write_prepare(bb->mem, bb->mem_size); memset(bb->mem, 0, bb->mem_size); }
    LOG("[nvn] DLSS network %llu: %dx%d, method %d (input 1/%d), preset %d, flags %#x, %d evaluation entries: a spatial upscale stands in (no NVIDIA network on the host)",
        (unsigned long long)o->serial, bb->w, bb->h, bb->method, dlss_factor(bb->method), bb->preset, bb->flags, bb->max_entries);
    return 1;
}
NVN_NOPROTO(nvnDlssNetworkFinalize, void, void *n) { dlss_net_t *o = n; OLOG("DLSS network %llu finalized after %llu evaluations, %llu history resets", (unsigned long long)o->serial, (unsigned long long)o->evaluations, (unsigned long long)o->resets); o->serial = 0; }
// sdk+0x6599d0: a new output resolution no larger than the one the network was made with (else false, the network unchanged)
NVN_NOPROTO(nvnDlssNetworkResize, NVNboolean, void *n, int w, int h) {
    dlss_net_t *o = n;
    if (w > o->b.w || h > o->b.h) return 0;
    o->out_w = w; o->out_h = h;
    return 1;
}
NVN_NOPROTO(nvnDlssParametersSetDefaults, void, void *p) {
    dlss_params_t *o = p; memset(o, 0, sizeof *o);
    o->mv_scale[0] = o->mv_scale[1] = 1; o->pre_exposure = 1;
    o->overlay_debug_layer = o->overlay_full_screen = o->overlay_show_nans = o->overlay_jitter_debug = ~0u;
}
NVN_NOPROTO(nvnDlssParametersSetInputColor, void, void *p, const NVNtexture *t, const NVNtextureView *v) { ((dlss_params_t *)p)->color = t; ((dlss_params_t *)p)->color_view = v; }
NVN_NOPROTO(nvnDlssParametersSetOutputColor, void, void *p, const NVNtexture *t, const NVNtextureView *v) { ((dlss_params_t *)p)->output = t; ((dlss_params_t *)p)->output_view = v; }
NVN_NOPROTO(nvnDlssParametersSetMotionVectors, void, void *p, const NVNtexture *t, const NVNtextureView *v) { ((dlss_params_t *)p)->motion = t; ((dlss_params_t *)p)->motion_view = v; }
NVN_NOPROTO(nvnDlssParametersSetDepth, void, void *p, const NVNtexture *t, const NVNtextureView *v) { ((dlss_params_t *)p)->depth = t; ((dlss_params_t *)p)->depth_view = v; }
NVN_NOPROTO(nvnDlssParametersSetBiasCurrentColorMask, void, void *p, const NVNtexture *t, const NVNtextureView *v) { ((dlss_params_t *)p)->bias = t; ((dlss_params_t *)p)->bias_view = v; }
NVN_NOPROTO(nvnDlssParametersSetExposureTexture, void, void *p, const NVNtexture *t, const NVNtextureView *v) { ((dlss_params_t *)p)->exposure = t; ((dlss_params_t *)p)->exposure_view = v; }
NVN_NOPROTO(nvnDlssParametersSetJitterOffset, void, void *p, float x, float y) { ((dlss_params_t *)p)->jitter[0] = x; ((dlss_params_t *)p)->jitter[1] = y; }
NVN_NOPROTO(nvnDlssParametersGetJitterOffset, void, const void *p, float *x, float *y) { if (x) *x = ((const dlss_params_t *)p)->jitter[0]; if (y) *y = ((const dlss_params_t *)p)->jitter[1]; }
NVN_NOPROTO(nvnDlssParametersSetMotionVectorScale, void, void *p, float x, float y) { ((dlss_params_t *)p)->mv_scale[0] = x; ((dlss_params_t *)p)->mv_scale[1] = y; }
NVN_NOPROTO(nvnDlssParametersGetMotionVectorScale, void, const void *p, float *x, float *y) { if (x) *x = ((const dlss_params_t *)p)->mv_scale[0]; if (y) *y = ((const dlss_params_t *)p)->mv_scale[1]; }
NVN_NOPROTO(nvnDlssParametersSetInputSubrectangleSize, void, void *p, int x, int y) { ((dlss_params_t *)p)->in_size[0] = x; ((dlss_params_t *)p)->in_size[1] = y; }
NVN_NOPROTO(nvnDlssParametersGetInputSubrectangleSize, void, const void *p, int *x, int *y) { if (x) *x = ((const dlss_params_t *)p)->in_size[0]; if (y) *y = ((const dlss_params_t *)p)->in_size[1]; }
NVN_NOPROTO(nvnDlssParametersSetInputColorSubrectangleBase, void, void *p, int x, int y) { ((dlss_params_t *)p)->color_base[0] = x; ((dlss_params_t *)p)->color_base[1] = y; }
NVN_NOPROTO(nvnDlssParametersGetInputColorSubrectangleBase, void, const void *p, int *x, int *y) { if (x) *x = ((const dlss_params_t *)p)->color_base[0]; if (y) *y = ((const dlss_params_t *)p)->color_base[1]; }
NVN_NOPROTO(nvnDlssParametersSetDepthSubrectangleBase, void, void *p, int x, int y) { ((dlss_params_t *)p)->depth_base[0] = x; ((dlss_params_t *)p)->depth_base[1] = y; }
NVN_NOPROTO(nvnDlssParametersGetDepthSubrectangleBase, void, const void *p, int *x, int *y) { if (x) *x = ((const dlss_params_t *)p)->depth_base[0]; if (y) *y = ((const dlss_params_t *)p)->depth_base[1]; }
NVN_NOPROTO(nvnDlssParametersSetMotionVectorSubrectangleBase, void, void *p, int x, int y) { ((dlss_params_t *)p)->mv_base[0] = x; ((dlss_params_t *)p)->mv_base[1] = y; }
NVN_NOPROTO(nvnDlssParametersGetMotionVectorSubrectangleBase, void, const void *p, int *x, int *y) { if (x) *x = ((const dlss_params_t *)p)->mv_base[0]; if (y) *y = ((const dlss_params_t *)p)->mv_base[1]; }
NVN_NOPROTO(nvnDlssParametersSetBiasCurrentColorSubrectangleBase, void, void *p, int x, int y) { ((dlss_params_t *)p)->bias_base[0] = x; ((dlss_params_t *)p)->bias_base[1] = y; }
NVN_NOPROTO(nvnDlssParametersGetBiasCurrentColorSubrectangleBase, void, const void *p, int *x, int *y) { if (x) *x = ((const dlss_params_t *)p)->bias_base[0]; if (y) *y = ((const dlss_params_t *)p)->bias_base[1]; }
NVN_NOPROTO(nvnDlssParametersSetOutputSubrectangleBase, void, void *p, int x, int y) { ((dlss_params_t *)p)->out_base[0] = x; ((dlss_params_t *)p)->out_base[1] = y; }
NVN_NOPROTO(nvnDlssParametersGetOutputSubrectangleBase, void, const void *p, int *x, int *y) { if (x) *x = ((const dlss_params_t *)p)->out_base[0]; if (y) *y = ((const dlss_params_t *)p)->out_base[1]; }
NVN_NOPROTO(nvnDlssParametersSetPreExposure, void, void *p, float e) { ((dlss_params_t *)p)->pre_exposure = e; }
NVN_NOPROTO(nvnDlssParametersGetPreExposure, float, const void *p) { return ((const dlss_params_t *)p)->pre_exposure; }
NVN_NOPROTO(nvnDlssParametersSetHistoryReset, void, void *p, uint32_t on) { ((dlss_params_t *)p)->reset = (uint8_t)B(on); }
NVN_NOPROTO(nvnDlssParametersGetHistoryReset, NVNboolean, const void *p) { return ((const dlss_params_t *)p)->reset; }
NVN_NOPROTO(nvnDlssParametersSetEvaluationIndex, void, void *p, int i) { ((dlss_params_t *)p)->eval_index = i; }
NVN_NOPROTO(nvnDlssParametersGetEvaluationIndex, int, const void *p) { return ((const dlss_params_t *)p)->eval_index; }
// NVIDIA's debug overlays drawn over the output: kept, and drawn by nothing on the host
NVN_NOPROTO(nvnDlssParametersSetOverlayDebugLayer, void, void *p, uint32_t v) { ((dlss_params_t *)p)->overlay_debug_layer = v; }
NVN_NOPROTO(nvnDlssParametersSetOverlayFullScreen, void, void *p, uint32_t v) { ((dlss_params_t *)p)->overlay_full_screen = v; }
NVN_NOPROTO(nvnDlssParametersSetOverlayShowNans, void, void *p, uint32_t v) { ((dlss_params_t *)p)->overlay_show_nans = v; }
NVN_NOPROTO(nvnDlssParametersSetOverlayJitterDebug, void, void *p, uint32_t v) { ((dlss_params_t *)p)->overlay_jitter_debug = v; }
// An evaluation (sdk+0x65a4e0 hands the parameters to NVIDIA's network): the replay resamples the input colour's subrectangle into the output's, at the evaluation's
// jitter (vka_dlss.c). As the driver: an input size of 0x0 is the input colour texture's size, the output's size is the network's output resolution and its base
// counts only with flag bit 28. The network keeps no history on the host, so a history reset only counts; the motion vectors, depth, bias mask and exposure feed
// NVIDIA's temporal network alone. The pre-exposure stays out of it too: DLSS's output is in its input's (pre-exposed) range
NVN_NOPROTO(nvnCommandBufferEvaluateDlss, void, NVNcommandBuffer *c, void *n, const void *p) {
    dlss_net_t *net = n; const dlss_params_t *o = p;
    if (!o || !o->color || !o->output) { ONCE_LOG("[RED] EvaluateDlss without an input or output colour texture: nothing evaluated"); return; }
    p_dlss *r = REC_SNAP(c, OP_DLSS, p_dlss, 2);
    r->color = tex_snap(r, SNAP_OFF(p_dlss), 0, o->color, nvn_caller()); copy_view(&r->cv, &r->has_cv, o->color_view);
    r->output = tex_snap(r, SNAP_OFF(p_dlss), 1, o->output, nvn_caller()); copy_view(&r->ov, &r->has_ov, o->output_view);
    const int no_size = !o->in_size[0] && !o->in_size[1];
    r->in_size[0] = no_size ? (int32_t)((const tex_t *)o->color)->d.width : o->in_size[0];
    r->in_size[1] = no_size ? (int32_t)((const tex_t *)o->color)->d.height : o->in_size[1];
    memcpy(r->in_base, o->color_base, sizeof r->in_base);
    if (net->b.flags & DLSS_FLAG_OUTPUT_BASE) memcpy(r->out_base, o->out_base, sizeof r->out_base);
    r->out_size[0] = net->out_w; r->out_size[1] = net->out_h;
    memcpy(r->jitter, o->jitter, sizeof r->jitter);
    r->reset = o->reset != 0; r->net = net->serial;
    net->evaluations++; if (o->reset) net->resets++;
}

// The dirty tile map parameters (nvnDirtyTileMapParametersSetDefaults sdk+0x64fd60 and its setters): 0x74 bytes. The driver hands them to the GPU as they are; what
// the console's GPU computes from them is not in the driver (below)
typedef struct {
    const NVNtexture *tex; const NVNtextureView *view;   // 0x00
    NVNcopyRegion region;                                 // 0x10
    int32_t input_select;     // 0x28 SetInputs' second value: the GPU command's word at +0x44, which a depth / stencil texture's mask replaces (sdk+0x64fc58)
    int32_t color_count;      // 0x2c SetConstantColorTable's second value (given to the GPU only for byte maps)
    uint64_t out;             // 0x30 the map's buffer address
    int32_t out_row_pitch, out_layer_stride;   // 0x38 (0: from the region and the tile size)
    uint64_t out_extra;       // 0x40 SetOutputs' last value
    uint64_t color_table;     // 0x48
    uint64_t color_extra;     // 0x50 SetConstantColorTable's last value
    int32_t color_stride;     // 0x58 its third (1 by default)
    int32_t tile_w, tile_h;   // 0x5c SetInputs' third and fourth (16 x 16 by default)
    uint32_t input_flags;     // 0x64 SetInputs' first: bit 0 a bit per tile instead of a byte
    int32_t depth_mask, stencil_mask;   // 0x68
    uint8_t expand[4];        // 0x70 SetExpandFootprint: four booleans, each multiplies a side by the texture's sample or block size (sdk+0x64fcac)
} dirty_tile_params_t;
_Static_assert(sizeof(dirty_tile_params_t) == 0x78 && offsetof(dirty_tile_params_t, input_select) == 0x28 && offsetof(dirty_tile_params_t, out) == 0x30 &&
               offsetof(dirty_tile_params_t, color_stride) == 0x58 && offsetof(dirty_tile_params_t, input_flags) == 0x64 && offsetof(dirty_tile_params_t, expand) == 0x70, "the driver's dirty tile map parameters");
enum { DIRTY_TILE_PARAMS_BYTES = 0x74 };   // what the driver's SetDefaults writes (the struct's tail is padding)
enum { DIRTY_TILE_BITS = 1 };
static uint64_t guest_of(const void *host) { return host ? nvn_direct_guest_of((uint64_t)(uintptr_t)host) : 0; }
NVN_NOPROTO(nvnDirtyTileMapParametersSetDefaults, void, void *p) {
    dirty_tile_params_t *o = p; memset(o, 0, DIRTY_TILE_PARAMS_BYTES);
    o->color_stride = 1; o->tile_w = o->tile_h = 16;
}
NVN_NOPROTO(nvnDirtyTileMapParametersSetTexture, void, void *p, const NVNtexture *t, const NVNtextureView *v, const NVNcopyRegion *r) {
    dirty_tile_params_t *o = p; o->tex = t; o->view = v; if (r) o->region = *r;
}
NVN_NOPROTO(nvnDirtyTileMapParametersGetTexture, void, const void *p, uint64_t *t, uint64_t *v, NVNcopyRegion *r) {
    const dirty_tile_params_t *o = p;
    if (t) *t = guest_of(o->tex);
    if (v) *v = guest_of(o->view);
    if (r) *r = o->region;
}
NVN_NOPROTO(nvnDirtyTileMapParametersSetInputs, void, void *p, uint32_t flags, int select, int tile_w, int tile_h) {
    dirty_tile_params_t *o = p; o->input_flags = flags; o->input_select = select; o->tile_w = tile_w; o->tile_h = tile_h;
}
NVN_NOPROTO(nvnDirtyTileMapParametersGetInputs, void, const void *p, uint32_t *flags, int *select, int *tile_w, int *tile_h) {
    const dirty_tile_params_t *o = p;
    if (flags) *flags = o->input_flags;
    if (select) *select = o->input_select;
    if (tile_w) *tile_w = o->tile_w;
    if (tile_h) *tile_h = o->tile_h;
}
NVN_NOPROTO(nvnDirtyTileMapParametersSetOutputs, void, void *p, uint64_t out, int row_pitch, int layer_stride, uint64_t extra) {
    dirty_tile_params_t *o = p; o->out = out; o->out_row_pitch = row_pitch; o->out_layer_stride = layer_stride; o->out_extra = extra;
}
NVN_NOPROTO(nvnDirtyTileMapParametersGetOutputs, void, const void *p, uint64_t *out, int *row_pitch, int *layer_stride, uint64_t *extra) {
    const dirty_tile_params_t *o = p;
    if (out) *out = o->out;
    if (row_pitch) *row_pitch = o->out_row_pitch;
    if (layer_stride) *layer_stride = o->out_layer_stride;
    if (extra) *extra = o->out_extra;
}
NVN_NOPROTO(nvnDirtyTileMapParametersSetConstantColorTable, void, void *p, uint64_t table, int count, int stride, uint64_t extra) {
    dirty_tile_params_t *o = p; o->color_table = table; o->color_count = count; o->color_stride = stride; o->color_extra = extra;
}
NVN_NOPROTO(nvnDirtyTileMapParametersGetConstantColorTable, void, const void *p, uint64_t *table, int *count, int *stride, uint64_t *extra) {
    const dirty_tile_params_t *o = p;
    if (table) *table = o->color_table;
    if (count) *count = o->color_count;
    if (stride) *stride = o->color_stride;
    if (extra) *extra = o->color_extra;
}
NVN_NOPROTO(nvnDirtyTileMapParametersSetDepthStencilMask, void, void *p, int depth, int stencil) { ((dirty_tile_params_t *)p)->depth_mask = depth; ((dirty_tile_params_t *)p)->stencil_mask = stencil; }
NVN_NOPROTO(nvnDirtyTileMapParametersGetDepthStencilMask, void, const void *p, int *depth, int *stencil) {
    if (depth) *depth = ((const dirty_tile_params_t *)p)->depth_mask;
    if (stencil) *stencil = ((const dirty_tile_params_t *)p)->stencil_mask;
}
NVN_NOPROTO(nvnDirtyTileMapParametersSetExpandFootprint, void, void *p, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    uint8_t *e = ((dirty_tile_params_t *)p)->expand; e[0] = (uint8_t)a; e[1] = (uint8_t)b; e[2] = (uint8_t)c; e[3] = (uint8_t)d;
}
NVN_NOPROTO(nvnDirtyTileMapParametersGetExpandFootprint, void, const void *p, uint8_t *a, uint8_t *b, uint8_t *c, uint8_t *d) {
    const uint8_t *e = ((const dirty_tile_params_t *)p)->expand;
    if (a) *a = e[0];
    if (b) *b = e[1];
    if (c) *c = e[2];
    if (d) *d = e[3];
}
// The map's shape (sdk+0x64faa0): tiles of tile_w x tile_h texels over the region, a byte per tile, or a bit per tile (input flag bit 0, the lowest bit the first
// tile); rows out_row_pitch bytes apart (0: just the row's bytes), and for a region of 2 or more layers, layers out_layer_stride apart (0: just the layer's rows)
typedef struct { uint32_t row_bytes, rows, layers, pitch, stride; uint8_t last; } dirty_tile_shape;
static int dirty_tile_shape_of(const dirty_tile_params_t *o, dirty_tile_shape *s) {
    if (o->tile_w <= 0 || o->tile_h <= 0 || o->region.width <= 0 || o->region.height <= 0) return 0;
    const uint32_t bits = o->input_flags & DIRTY_TILE_BITS, tiles = ((uint32_t)o->region.width + (uint32_t)o->tile_w - 1) / (uint32_t)o->tile_w;
    s->row_bytes = bits ? (tiles + 7) / 8 : tiles;
    s->last = bits && tiles % 8 ? (uint8_t)((1u << tiles % 8) - 1) : 0xff;
    s->rows = ((uint32_t)o->region.height + (uint32_t)o->tile_h - 1) / (uint32_t)o->tile_h;
    s->pitch = o->out_row_pitch ? (uint32_t)o->out_row_pitch : s->row_bytes;
    s->layers = o->region.depth >= 2 ? (uint32_t)o->region.depth : 1;
    s->stride = s->layers > 1 ? (o->out_layer_stride ? (uint32_t)o->out_layer_stride : s->rows * s->pitch) : 0;
    return 1;
}
// The console's GPU computes the map in a command of its own (the driver records opcode 0x5b with these parameters and the texture's internals, sdk+0x64fad0), whose
// test for a dirty tile is not in the driver. The host marks every tile of the region dirty (each byte 0xff, or each tile's bit set): a reader that skips the clean
// tiles then redoes them all, which costs work but never loses any. In command order, through inline buffer updates (the bytes between rows are left as they are)
static void dirty_tile_map(NVNcommandBuffer *c, const dirty_tile_params_t *o) {
    ONCE_LOG("[RED] ComputeDirtyTileMap: the host marks every tile dirty (what the console's GPU tests is unknown)");
    dirty_tile_shape s;
    if (!o->out || !dirty_tile_shape_of(o, &s)) { ONCE_LOG("[RED] ComputeDirtyTileMap with no output, an empty region or a zero tile size: nothing written"); return; }
    const uint64_t out = nvn_direct_host_of(o->out);
    const int packed = s.pitch == s.row_bytes && (s.layers == 1 || s.stride == s.rows * s.pitch);
    const size_t n = packed ? (size_t)s.row_bytes * s.rows * s.layers : s.row_bytes;
    uint8_t *bytes = malloc(n); if (!bytes) die("dirty tile map allocation failed");
    memset(bytes, 0xff, n);
    for (size_t r = 0; r < n / s.row_bytes; r++) bytes[(r + 1) * s.row_bytes - 1] = s.last;
    if (packed) h_nvnCommandBufferUpdateUniformBuffer(c, out, n, 0, n, bytes);
    else for (uint32_t l = 0; l < s.layers; l++) for (uint32_t r = 0; r < s.rows; r++) h_nvnCommandBufferUpdateUniformBuffer(c, out + (uint64_t)l * s.stride + (uint64_t)r * s.pitch, n, 0, n, bytes);
    free(bytes);
}
NVN_NOPROTO(nvnCommandBufferComputeDirtyTileMapEx, void, NVNcommandBuffer *c, const void *p) { dirty_tile_map(c, p); }
// sdk+0x64fa30: Ex with the defaults bar the texture, the second input and the outputs, a bit per tile (flags 0x80000001) and 16 x 16 tiles
NVN_NOPROTO(nvnCommandBufferComputeDirtyTileMap, void, NVNcommandBuffer *c, const NVNtexture *t, const NVNtextureView *v, const NVNcopyRegion *r, int select, uint64_t out, int row_pitch, int layer_stride) {
    dirty_tile_params_t o; memset(&o, 0, sizeof o);
    o.tex = t; o.view = v; if (r) o.region = *r;
    o.input_select = select; o.out = out; o.out_row_pitch = row_pitch; o.out_layer_stride = layer_stride;
    o.tile_w = o.tile_h = 16; o.input_flags = 0x80000001u;
    dirty_tile_map(c, &o);
}

// Cooperative-vector matrices: the layouts of the console's GPU, ported from its driver (nvnDeviceGetCooperativeVectorMatrixSize sdk+0x651a30, the CPU conversion
// nvnDeviceConvertCooperativeVectorMatrix sdk+0x6510c0 and its helpers sdk+0x6571b0 / +0x657300 / +0x657420), so a matrix comes out as the console's would.
// Component types: 0 fp16, 1 fp32, 2 8-bit, 3 32-bit, 4 8-bit, 5 32-bit (the integer ones copy as they are), 8 fp8 E4M3, 9 fp8 E5M2. Layouts: 0 row-major,
// 1 column-major (stride bytes between rows / columns), 2 inferencing-optimal, 3 training-optimal (16-byte chunks of 8 rows in 128-byte blocks, 4 blocks to 512 bytes,
// training's with 512 bytes after each such group)
static const uint32_t g_cv_shift[6] = {1, 2, 0, 2, 0, 2};   // log2 of the component's bytes (sdk+0xc8065c)
static uint32_t cv_shift(uint32_t type) { return type <= 5 ? g_cv_shift[type] : 0; }
static uint32_t cv_udiv(uint32_t a, uint32_t b) { return b ? a / b : 0; }   // (AArch64's: x / 0 = 0)
// fp32 has no optimal layout: it is row-major with rows 16 bytes aligned; fp8 in an optimal layout is stored as fp16
static void cv_effective(const coopvec_desc *d, uint32_t cols, coopvec_desc *e) {
    *e = *d;
    if ((d->layout & ~1u) != 2) return;
    if ((d->type & ~1u) == 8) e->type = 0;
    else if (d->type == 1) { e->layout = 0; e->stride = ((uint64_t)cols * 4 + 15) & 0x7fffffff0ull; }
}
static uint32_t cv_tail(uint32_t blocks) { uint32_t t = blocks & 3; return t == 0 || t == 3 ? 4 : t; }   // the blocks in the last group
static uint64_t cv_bytes(const coopvec_desc *e, uint32_t rows, uint32_t cols) {   // one copy
    if (!rows || !cols) return 0;
    const uint32_t sh = cv_shift(e->type);
    if (e->layout == 0) return e->stride * (rows - 1) + (uint32_t)(cols << sh);
    if (e->layout == 1) return e->stride * (cols - 1) + (uint32_t)(rows << sh);
    if (e->layout != 2 && e->layout != 3) return 0;
    uint32_t blocks = ((cols + (16u >> sh) - 1) >> (4 - sh)) * ((rows + 7) >> 3);
    if ((blocks & 3) == 3) blocks++;
    uint32_t n = blocks << 7;
    if (e->layout == 3) n += ((n + 0x180) & ~0x1ffu) - 0x200;
    return n;
}
// 8-bit components in the inferencing-optimal layout take two copies, the second with columns swizzled (cv_swizzle)
static int cv_copies(const coopvec_desc *e) { return 1 + (e->layout == 2 && (e->type == 2 || e->type == 4 || e->type > 5)); }
static uint32_t cv_swizzle(uint32_t c) { return (c & ~0xeu) | ((c >> 1) & 6) | ((c >> 1) & 1) << 3; }
// The row and column of the component at byte off of a matrix (0: padding)
static int cv_position(const coopvec_desc *e, uint32_t rows, uint32_t cols, uint32_t off, int copy, uint32_t *row, uint32_t *col) {
    const uint32_t sh = cv_shift(e->type), stride = (uint32_t)e->stride;
    if (e->layout == 0 || e->layout == 1) {
        const uint32_t q = cv_udiv(off, stride), in = (off - q * stride) >> sh;
        *row = e->layout ? in : q; *col = e->layout ? q : in;
        return *row < rows && *col < cols;
    }
    if (e->layout != 2 && e->layout != 3) return 0;
    const uint32_t chunks = (cols + (16u >> sh) - 1) >> (4 - sh), blocks = chunks * ((rows + 7) >> 3);
    if (e->layout == 3) { if (off & 0x200) return 0; off -= (off >> 10) << 9; }
    const uint32_t group = off >> 9, in = off & 0x1ff, gsize = group == blocks >> 2 ? cv_tail(blocks) : 4;
    const uint32_t blk = ((gsize - 1) & (in >> 2)) + group * 4, word = in >> ((gsize >> 1) + 2), rb = cv_udiv(blk, chunks), ch = blk - rb * chunks;
    *row = (word >> 2) + rb * 8;
    *col = (word & 3) * (4u >> sh) + ((off & 3) >> sh) + (ch << (4 - sh));
    return *row < rows && (copy ? cv_swizzle(*col) : *col) < cols;
}
// The byte offset of a component of a matrix
static uint64_t cv_offset(const coopvec_desc *e, uint32_t rows, uint32_t cols, uint32_t row, uint32_t col, int copy) {
    const uint32_t sh = cv_shift(e->type);
    if (copy) col = cv_swizzle(col);
    if (e->layout == 0) return (uint32_t)(col << sh) + e->stride * row;
    if (e->layout == 1) return (uint32_t)(row << sh) + e->stride * col;
    if (e->layout != 2 && e->layout != 3) return 0;
    const uint32_t per = 16u >> sh, chunks = (cols + per - 1) >> (4 - sh), blocks = chunks * ((rows + 7) >> 3);
    const uint32_t blk = chunks * (row >> 3) + (col >> (4 - sh)), word = ((col & (per - 1)) >> (2 - sh)) + ((row & 7) << 2);
    const uint32_t group = blk >> 2, gsize = group == blocks >> 2 ? cv_tail(blocks) : 4;
    return (uint64_t)(group << 9) + ((col & ((4u >> sh) - 1)) << sh) + (e->layout == 3 ? group << 9 : 0) + (gsize * word + (blk & 3)) * 4;
}
// fp16, E4M3 and E5M2 (mantissa and exponent bits); E4M3 has no infinity and one NaN pattern (0x7f / 0xff)
static int cv_float_bits(uint32_t type, uint32_t *m, uint32_t *e) {
    if (type == 0) { *m = 10; *e = 5; return 1; }
    if (type == 8) { *m = 3; *e = 4; return 1; }
    if (type == 9) { *m = 2; *e = 5; return 1; }
    return 0;
}
static float cv_load(const uint8_t *p, uint32_t type) {
    float f; uint32_t m, e;
    if (type == 1) { memcpy(&f, p, 4); return f; }
    if (!cv_float_bits(type, &m, &e)) return 0;
    const uint32_t v = type ? p[0] : (uint32_t)p[0] | (uint32_t)p[1] << 8, sign = (v >> (m + e)) & 1, emax = (1u << e) - 1;
    uint32_t mant = v & ((1u << m) - 1), ex = (v >> m) & emax, bits;
    if (!(v & ~(1u << (m + e)))) return 0;   // either zero: +0
    if (type == 8 && (v & 0x7f) == 0x7f) bits = 0x7fffffffu;
    else if (type != 8 && ex == emax) bits = 0x7f800000u | (mant ? 0x7fffffu : 0);
    else {
        int x = (int)ex;
        if (!ex) { x = 1; while (!(mant & (1u << m))) { mant <<= 1; x--; } mant &= ~(1u << m); }
        bits = (uint32_t)(x - (int)(emax >> 1) + 127) << 23 | mant << (23 - m);
    }
    bits |= sign << 31;
    memcpy(&f, &bits, 4); return f;
}
// Rounded to nearest even; past the largest exponent: infinity (E4M3: NaN, unless the exponent is just its top one); a NaN: the positive all-ones pattern; values
// below half the smallest denormal's mantissa range: signed zero
static void cv_store(uint8_t *p, float f, uint32_t type) {
    uint32_t m, e;
    if (type == 1) { memcpy(p, &f, 4); return; }
    if (!cv_float_bits(type, &m, &e)) return;
    uint32_t bits; memcpy(&bits, &f, 4);
    const uint32_t emax = (1u << e) - 1, bias = (1u << (e - 1)) - 1, fe = bits >> 23 & 0xff, fm = bits & 0x7fffff;
    uint32_t sign = bits >> 31, E, M, zero = f == 0.0f;
    if (fe == 0xff) { E = emax; M = 0; if (type == 8 || fm) { sign = 0; M = (1u << m) - 1; } }
    else {
        const int x = (int)fe - 127 + (int)bias;
        if (x > 0) {
            uint32_t r = fm + ((1u << (22 - m)) - 1) + (fm >> (23 - m) & 1);
            E = (uint32_t)x; if (r & 0x800000) { r = 0; E++; }
            M = r >> (23 - m);
        } else if ((uint32_t)-x > m) { E = 0; M = 0; zero = 1; }
        else {
            const uint32_t full = fm | 0x800000, shift = 24 - m - (uint32_t)x;   // (23 - m) + (1 - x)
            E = 0; M = (full + ((1u << (shift - 1)) - 1) + (full >> shift & 1)) >> shift;
        }
        if (E >= emax) {
            if (type != 8) { E = emax; M = 0; }
            else if (E > emax) { E = emax; M = (1u << m) - 1; sign = 0; }
        }
    }
    const uint32_t v = zero ? sign << (m + e) : sign << (m + e) | E << m | M;
    if (type == 0) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); } else p[0] = (uint8_t)v;
}
void nvn_coopvec_convert(const uint8_t *src, const coopvec_desc *sd, uint8_t *dst, const coopvec_desc *dd, uint32_t rows, uint32_t cols) {
    coopvec_desc s, d; cv_effective(sd, cols, &s); cv_effective(dd, cols, &d);
    const int src_fp8 = (sd->type & ~1u) == 8 && (sd->layout & ~1u) == 2, dst_fp8 = (dd->type & ~1u) == 8 && (dd->layout & ~1u) == 2;
    const int convert = s.type != d.type || src_fp8 || dst_fp8, copies = cv_copies(&d);
    const uint64_t n = cv_bytes(&d, rows, cols);
    const uint32_t ssize = 1u << cv_shift(s.type), dsize = 1u << cv_shift(d.type);
    if (!n) return;
    dm_guest_write_prepare(dst, n * (uint64_t)copies);
    for (int k = 0; k < copies; k++) {
        uint8_t *const out = dst + n * (uint64_t)k;
        for (uint64_t off = 0; off < n; off += dsize) {
            uint32_t row, col;
            if (!cv_position(&d, rows, cols, (uint32_t)off, k, &row, &col)) { memset(out + off, 0, dsize); continue; }
            const uint8_t *const in = src + cv_offset(&s, rows, cols, row, col, k);
            if (!convert) { memcpy(out + off, in, ssize); continue; }
            float f = cv_load(in, s.type);
            if (dst_fp8) { uint8_t q[4] = {0}; cv_store(q, f, dd->type); f = cv_load(q, dd->type); }   // fp8's precision, kept as fp16
            cv_store(out + off, f, d.type);
        }
    }
}
NVN_NOPROTO(nvnDeviceGetCooperativeVectorMatrixSize, size_t, const NVNdevice *dev, const void *desc, int rows, int cols) {
    (void)dev; coopvec_desc e; cv_effective(desc, (uint32_t)cols, &e);
    return (size_t)(cv_bytes(&e, (uint32_t)rows, (uint32_t)cols) * (uint64_t)cv_copies(&e));
}
// The sizes are the caller's; the driver reads neither and writes the destination's whole size
NVN_NOPROTO(nvnDeviceConvertCooperativeVectorMatrix, NVNboolean, const NVNdevice *dev, const void *src, size_t src_size, const void *src_desc, void *dst, size_t dst_size, const void *dst_desc, int rows, int cols) {
    (void)dev; (void)src_size; (void)dst_size;
    nvn_coopvec_convert(src, src_desc, dst, dst_desc, (uint32_t)rows, (uint32_t)cols);
    return 1;
}
// sdk+0x64df10 records it for the GPU (opcode 0x6e); the replay runs the same CPU conversion once the GPU has done the submission's work before it (vka_exec.c)
NVN_NOPROTO(nvnCommandBufferConvertCooperativeVectorMatrix, void, NVNcommandBuffer *c, NVNbufferAddress src, size_t src_size, const void *src_desc, NVNbufferAddress dst, size_t dst_size, const void *dst_desc, int rows, int cols) {
    (void)src_size; (void)dst_size;
    p_coopvec_convert *r = REC(c, OP_COOPVEC_CONVERT, p_coopvec_convert);
    r->src = src; r->dst = dst; r->sd = *(const coopvec_desc *)src_desc; r->dd = *(const coopvec_desc *)dst_desc; r->rows = rows; r->cols = cols;
}

// ---------- Mesh draws, texture clears / resolves / readback, descriptor table writes, views ----------
// Signatures and meanings of the NVN2-only calls are derived from the NVN2 driver in OoT's sdk NSO (functions at sdk+0x..., nvn_protos.h)
#if defined(__ANDROID__)
// The other mesh draws, recorded as DrawMeshTasks is (nvn_obj.h NVN_PRIM_MESH / NVN_PRIM_MESH_3D) and drawn with vkCmdDrawMeshTasks*EXT (vka_draw.c). An indirect draw's
// records are written by the GPU earlier in the frame: {count, first} for the one-dimensional draws (the words DrawMeshTasks writes, sdk+0x681470), {x, y, z} for the 3D ones
NVN(nvnCommandBufferDrawMeshTasksIndirect, void, NVNcommandBuffer *c, NVNbufferAddress ind) { rec_draw(c, (p_draw){NVN_PRIM_MESH, 0, 0, 1, 0, ind}); }
NVN(nvnCommandBufferDrawMeshTasks3D, void, NVNcommandBuffer *c, int x, int y, int z) { rec_draw(c, (p_draw){NVN_PRIM_MESH_3D, 0, x, y, z}); }
NVN(nvnCommandBufferDrawMeshTasks3DIndirect, void, NVNcommandBuffer *c, NVNbufferAddress ind) { rec_draw(c, (p_draw){NVN_PRIM_MESH_3D, 0, 0, 1, 0, ind}); }
// Multi-draws: n records stride bytes apart, at most the u32 at cnt of them for the Count forms; one record for the whole call
static void rec_mesh_multi(NVNcommandBuffer *c, int prim, NVNbufferAddress ind, NVNbufferAddress cnt, int n, ptrdiff_t stride) {
    if (n <= 0) return;
    if (stride <= 0 || stride > INT32_MAX) { ONCE_LOG("[RED] mesh multi-draw with stride %td: dropped", stride); return; }
    rec_draw(c, (p_draw){prim, 0, 0, 1, 0, ind, cnt, 0, n, (int32_t)stride});
}
NVN(nvnCommandBufferMultiDrawMeshTasksIndirect, void, NVNcommandBuffer *c, NVNbufferAddress ind, int n, ptrdiff_t stride) { rec_mesh_multi(c, NVN_PRIM_MESH, ind, 0, n, stride); }
NVN(nvnCommandBufferMultiDrawMeshTasksIndirectCount, void, NVNcommandBuffer *c, NVNbufferAddress ind, NVNbufferAddress cnt, int max, ptrdiff_t stride) { rec_mesh_multi(c, NVN_PRIM_MESH, ind, cnt, max, stride); }
NVN(nvnCommandBufferMultiDrawMeshTasks3DIndirect, void, NVNcommandBuffer *c, NVNbufferAddress ind, int n, ptrdiff_t stride) { rec_mesh_multi(c, NVN_PRIM_MESH_3D, ind, 0, n, stride); }
NVN(nvnCommandBufferMultiDrawMeshTasks3DIndirectCount, void, NVNcommandBuffer *c, NVNbufferAddress ind, NVNbufferAddress cnt, int max, ptrdiff_t stride) { rec_mesh_multi(c, NVN_PRIM_MESH_3D, ind, cnt, max, stride); }
#endif
// A clear with signed integer values: their bits, as ClearTextureui's (the replay's clear_tex puts them in the clear value as they are)
NVN(nvnCommandBufferClearTexturei, void, NVNcommandBuffer *c, const NVNtexture *t, const NVNtextureView *v, const NVNcopyRegion *r, const int *rgba, int mask) {
    p_clear_tex *o = REC_SNAP(c, OP_CLEAR_TEX, p_clear_tex, 1); o->tex = tex_snap(o, SNAP_OFF(p_clear_tex), 0, t, nvn_caller()); copy_view(&o->view, &o->has_view, v); o->r = *r; o->mask = mask; o->is_uint = 2; memcpy(o->c.u, rgba, 16);
}
// The tiled cache's multisample resolve (sdk+0x698a00: the plain form passes the view arguments as null into the view form's resolve): the same resolve as Downsample
NVN(nvnCommandBufferTiledDownsample, void, NVNcommandBuffer *c, const NVNtexture *s, const NVNtexture *d) { rec_downsample(c, s, NULL, d, NULL, nvn_caller()); }
NVN(nvnCommandBufferTiledDownsampleTextureView, void, NVNcommandBuffer *c, const NVNtexture *s, const NVNtextureView *sv, const NVNtexture *d, const NVNtextureView *dv) {
    rec_downsample(c, s, sv, d, dv, nvn_caller());
}
// The driver sends one method, DECOMPRESS_ZETA_SURFACE = 1 (sdk+0x684fa0: 3D method 0x11fc): the bound depth buffer's compressed tiles are expanded in place, its values unchanged.
// The host's depth images have no compression the game can see, so their content already is the expanded one
NVN(nvnCommandBufferResolveDepthBuffer, void, NVNcommandBuffer *c) { (void)c; }
// Whether two views select the same thing: the same parts set, each set part equal (sdk+0x66b0b0 compares the set bits first, then each set field). After nxnative (MIT)
NVN(nvnTextureViewCompare, NVNboolean, const NVNtextureView *a, const NVNtextureView *b) {
    const view_t *x = (const view_t *)a, *y = (const view_t *)b;
    return x->set_levels == y->set_levels && x->set_layers == y->set_layers && x->set_format == y->set_format && x->set_target == y->set_target && x->set_swizzle == y->set_swizzle && x->set_dsmode == y->set_dsmode
        && (!x->set_levels || (x->base_level == y->base_level && x->nlevels == y->nlevels)) && (!x->set_layers || (x->min_layer == y->min_layer && x->nlayers == y->nlayers))
        && (!x->set_format || x->format == y->format) && (!x->set_target || x->target == y->target) && (!x->set_swizzle || !memcmp(x->swz, y->swz, sizeof x->swz)) && (!x->set_dsmode || x->dsmode == y->dsmode);
}
NVN(nvnTextureBuilderSetGLTextureName, void, NVNtextureBuilder *b, uint32_t name) { TB(b)->gl_name = name; }   // (sdk+0x66a730 stores it for its getter)
// Raw descriptor words (sdk+0x66c760 / 0x6682e0): n bytes from words into the pool's memory at entry id's 32 bytes + off, as the driver copies them. The sampler pool's entries
// they reach are decoded as an Ampere TSC and registered as SamplerPoolRegisterSampler registers a sampler; texture headers (TIC) are not decoded: the IDs keep what was registered
static int desc_write(const dpool_t *p, int id, ptrdiff_t off, size_t n, const void *words, const char *kind, int *first, int *last) {
    const uint64_t at = (uint64_t)id * 32 + (uint64_t)off;
    if (id < 0 || off < 0 || (n & 3) || at + n > (uint64_t)p->n * 32 || !p->pool || !p->pool->mem) {
        ONCE_LOG("[RED] %sPoolWriteDescriptorTableData: id %d offset %td, %zu bytes outside the pool's %d entries (or no storage): not written", kind, id, off, n, p->n); return 0; }
    if (!n) return 0;
    uint8_t *dst = p->pool->mem + p->off + at;
    dm_guest_write_prepare(dst, n); memcpy(dst, words, n);
    *first = (int)(at / 32); *last = (int)((at + n - 1) / 32);
    return 1;
}
NVN(nvnTexturePoolWriteDescriptorTableData, void, const NVNtexturePool *tp, int id, ptrdiff_t off, size_t n, const void *words) {
    int first, last;
    if (desc_write((const dpool_t *)tp, id, off, n, words, "Texture", &first, &last))
        ONCE_LOG("[RED] TexturePoolWriteDescriptorTableData: raw texture headers are written to the pool's memory, not decoded (no Ampere TIC to NVN format table): IDs %d-%d keep what was registered", first, last);
}
static samp_par tsc_decode(const uint32_t w[8]) {   // an Ampere TSC entry (the base renderer's Tegra::Texture::TSCEntry)
    static const uint8_t wrap[8] = {NVN_WRAP_MODE_REPEAT, NVN_WRAP_MODE_MIRRORED_REPEAT, NVN_WRAP_MODE_CLAMP_TO_EDGE, NVN_WRAP_MODE_CLAMP_TO_BORDER, NVN_WRAP_MODE_CLAMP,
                                    NVN_WRAP_MODE_MIRROR_CLAMP_TO_EDGE, NVN_WRAP_MODE_MIRROR_CLAMP_TO_BORDER, NVN_WRAP_MODE_MIRROR_CLAMP};
    static const float aniso[8] = {1, 2, 4, 6, 8, 10, 12, 16};
    samp_par o; memset(&o, 0, sizeof o);
    o.ws = wrap[w[0] & 7]; o.wt = wrap[w[0] >> 3 & 7]; o.wr = wrap[w[0] >> 6 & 7];
    o.cmp_mode = w[0] >> 9 & 1 ? NVN_COMPARE_MODE_COMPARE_R_TO_TEXTURE : NVN_COMPARE_MODE_NONE; o.cmp_func = (uint8_t)((w[0] >> 10 & 7) + NVN_COMPARE_FUNC_NEVER);
    o.aniso = aniso[w[0] >> 20 & 7];
    const unsigned mag = w[1] & 3, min = w[1] >> 4 & 3, mip = w[1] >> 6 & 3;   // filters: 1 nearest, 2 linear; mip: 1 none, 2 nearest, 3 linear
    o.magf = mag == 2 ? NVN_MAG_FILTER_LINEAR : NVN_MAG_FILTER_NEAREST;
    o.minf = (uint8_t)((mip == 3 ? NVN_MIN_FILTER_NEAREST_MIPMAP_LINEAR : mip == 2 ? NVN_MIN_FILTER_NEAREST_MIPMAP_NEAREST : NVN_MIN_FILTER_NEAREST) + (min == 2 ? 1 : 0));
    o.reduction = (uint8_t)(w[1] >> 10 & 3);
    o.lod_bias = (float)((int32_t)(w[1] << 7) >> 19) / 256.0f;   // 13 bits from bit 12, signed
    o.lod_min = (float)(w[2] & 0xfff) / 256.0f; o.lod_max = (float)(w[2] >> 12 & 0xfff) / 256.0f;
    memcpy(o.border, &w[4], 16);
    return o;
}
NVN(nvnSamplerPoolWriteDescriptorTableData, void, const NVNsamplerPool *sp, int id, ptrdiff_t off, size_t n, const void *words) {
    const dpool_t *p = (const dpool_t *)sp; int first, last;
    if (!desc_write(p, id, off, n, words, "Sampler", &first, &last)) return;
    for (int i = first; i <= last; i++) {
        uint32_t w[8]; memcpy(w, p->pool->mem + p->off + (uint64_t)i * 32, 32);
        samp_par s = tsc_decode(w); mtl_reg_samp(dpool_base(g_spb, g_nspb, sp) + i, &s);
    }
}

// ---------- Command-buffer state (Lichii) ----------
// The calls nvn_protos.h has no prototype for take the signatures of the NVN2 driver in Ocarina of Time's sdk (the registers each reads, the methods it writes).
// The replay applies what Vulkan can express (vka_draw.c, vka_exec.c, vka_pso.c); what it cannot is logged once where it is used

// Tessellation: the patch size is part of the pipeline (vka_pso.c). The levels are NVN's defaults for a program without a tessellation control shader, which Vulkan
// cannot draw (it needs the control stage): kept in the replay's state, such draws are logged and skipped (vka_draw.c)
NVN(nvnCommandBufferSetPatchSize, void, NVNcommandBuffer *c, int n) { REC(c, OP_PATCH_SIZE, p_patch_size)->n = n; }
NVN(nvnCommandBufferSetInnerTessellationLevels, void, NVNcommandBuffer *c, const float *l) { p_tess_levels *r = REC(c, OP_TESS_LEVELS, p_tess_levels); r->outer = 0; memcpy(r->v, l, 2 * sizeof *l); }
NVN(nvnCommandBufferSetOuterTessellationLevels, void, NVNcommandBuffer *c, const float *l) { p_tess_levels *r = REC(c, OP_TESS_LEVELS, p_tess_levels); r->outer = 1; memcpy(r->v, l, 4 * sizeof *l); }
// Multisample coverage: the mask (the driver keeps its low 16 bits) is the pipeline's pSampleMask
NVN(nvnCommandBufferSetSampleMask, void, NVNcommandBuffer *c, int m) { REC(c, OP_SAMPLE_MASK, p_sample_mask)->mask = (uint32_t)m & 0xffff; }
// Conservative rasterization (VK_EXT_conservative_rasterization, overestimation): on / off and the dilation, which the driver rounds up to quarter pixels (at most 0.75).
// The mode's pre-snap variants and the subpixel precision bias have no Vulkan equivalent (logged once at a draw that uses them). SetConservativeRasterMode: mode in w1,
// a 4-bit value the driver writes beside it in w2, whose meaning is not known (kept out of the record)
NVN(nvnCommandBufferSetConservativeRasterEnable, void, NVNcommandBuffer *c, uint32_t en) { *REC(c, OP_RASTER, p_raster) = (p_raster){RASTER_CONSERVATIVE, B(en) != 0, 0, 0}; }
NVN(nvnCommandBufferSetConservativeRasterDilate, void, NVNcommandBuffer *c, float d) {
    int q = d > 0 ? (int)(d * 4.0f) + (d * 4.0f > (float)(int)(d * 4.0f)) : 0; if (q > 3) q = 3;
    *REC(c, OP_RASTER, p_raster) = (p_raster){RASTER_DILATE, 0, 0, (float)q / 4.0f};
}
NVN_NOPROTO(nvnCommandBufferSetConservativeRasterMode, void, NVNcommandBuffer *c, int mode, int extra) { (void)extra; *REC(c, OP_RASTER, p_raster) = (p_raster){RASTER_CONSERVATIVE_MODE, mode, 0, 0}; }
NVN(nvnCommandBufferSetSubpixelPrecisionBias, void, NVNcommandBuffer *c, int x, int y) { *REC(c, OP_RASTER, p_raster) = (p_raster){RASTER_SUBPIXEL_BIAS, x & 0x1f, y & 0x1f, 0}; }
// The provoking vertex of flat-shaded outputs: the driver's value is the GPU's (0 the first vertex, 1 the last); the last needs VK_EXT_provoking_vertex
NVN_NOPROTO(nvnCommandBufferSetProvokingVertexMode, void, NVNcommandBuffer *c, int mode) { *REC(c, OP_RASTER, p_raster) = (p_raster){RASTER_PROVOKING, mode, 0, 0}; }
// Clamping of blend inputs and results on float targets: Vulkan blends them unclamped, which is NVN's default
NVN_NOPROTO(nvnCommandBufferSetFloatBlendClamp, void, NVNcommandBuffer *c, uint32_t en) { (void)c; if (B(en)) ONCE_LOG("[RED] SetFloatBlendClamp(true): float targets blend unclamped on the host (Vulkan has no such clamp)"); }
// Representative fragment test: lets the GPU skip fragments of a primitive beyond a representative set; shading all of them is a valid result, which the host does
NVN_NOPROTO(nvnCommandBufferSetRepresentativeFragmentTest, void, NVNcommandBuffer *c, uint32_t en) { (void)c; (void)en; }
// GPU hints with no effect on what the commands produce: ZCull's compression of 32-bit float depth (the host does no ZCull, as for SaveZCullData), the share of the GPU's
// resources a kind of work may take (kind 0 / 1, a fraction) and its priority, and the tiled cache's subtile size
NVN_NOPROTO(nvnCommandBufferSetZCullZF32CompressionEnable, void, NVNcommandBuffer *c, uint32_t en) { (void)c; (void)en; }
NVN_NOPROTO(nvnCommandBufferSetResourceOccupancy, void, NVNcommandBuffer *c, int kind, float share) { (void)c; (void)kind; (void)share; }
NVN_NOPROTO(nvnCommandBufferSetGraphicsWorkloadPriority, void, NVNcommandBuffer *c, int prio) { (void)c; (void)prio; }
NVN_NOPROTO(nvnCommandBufferSetComputeWorkloadPriority, void, NVNcommandBuffer *c, int prio) { (void)c; (void)prio; }
NVN_NOPROTO(nvnCommandBufferOverrideSubtileSize, void, NVNcommandBuffer *c, uint32_t en, int size) { (void)c; (void)en; (void)size; }
// Control memory callbacks: the host records no control commands (cmd_t.ctl_size), so the callback this enables never becomes due
NVN_NOPROTO(nvnCommandBufferSetControlMemoryCallbackEnabled, void, NVNcommandBuffer *c, uint32_t on) { (void)c; (void)on; }
// Scratch memory for shaders' local memory: the translated shaders keep local memory per invocation (the recompiler sizes it from the shader header), so the replay reads
// none of it. The range must lie in the pool, as the driver needs
NVN(nvnCommandBufferSetShaderScratchMemory, void, NVNcommandBuffer *c, const NVNmemoryPool *p, ptrdiff_t off, size_t n) {
    (void)c; const pool_t *pl = (const pool_t *)p;
    if (pl && (off < 0 || (uint64_t)off + n > pl->size)) ONCE_LOG("[RED] SetShaderScratchMemory: %#zx bytes at %#llx are outside their pool (%#llx bytes)", n, (unsigned long long)off, (unsigned long long)pl->size);
}
// A wait for a sync object in the command stream: the replay waits for the sync's fence where the command runs (vka_exec.c), read then, as the GPU reads it
NVN(nvnCommandBufferWaitSync, void, NVNcommandBuffer *c, const NVNsync *s) { REC(c, OP_WAIT_SYNC, p_wait_sync)->sync = (const sync_t *)s; }
// Compute and copy enables: like SetRenderEnable(Conditional), for dispatches and for copies (mode: render if the counters at addr and addr + 16 are equal / not equal)
NVN_NOPROTO(nvnCommandBufferSetComputeEnable, void, NVNcommandBuffer *c, uint32_t en) { *REC(c, OP_ENABLE_COND, p_enable_cond) = (p_enable_cond){0, B(en) ? 2 : 3, 0}; }
NVN_NOPROTO(nvnCommandBufferSetComputeEnableConditional, void, NVNcommandBuffer *c, NVNconditionalRenderMode m, NVNbufferAddress a) { *REC(c, OP_ENABLE_COND, p_enable_cond) = (p_enable_cond){0, (int)m & 1, a}; }
NVN_NOPROTO(nvnCommandBufferSetCopyEnable, void, NVNcommandBuffer *c, uint32_t en) { *REC(c, OP_ENABLE_COND, p_enable_cond) = (p_enable_cond){1, B(en) ? 2 : 3, 0}; }
NVN_NOPROTO(nvnCommandBufferSetCopyEnableConditional, void, NVNcommandBuffer *c, NVNconditionalRenderMode m, NVNbufferAddress a) { *REC(c, OP_ENABLE_COND, p_enable_cond) = (p_enable_cond){1, (int)m & 1, a}; }
// Exclusive scissor (VK_NV_scissor_exclusive): fragments inside the rectangle are discarded
NVN_NOPROTO(nvnCommandBufferEnableExclusiveScissor, void, NVNcommandBuffer *c, uint32_t en) { p_excl_scissor *r = RECN(c, OP_EXCL_SCISSOR, p_excl_scissor, r, 0); r->what = 0; r->first = B(en) != 0; }
NVN_NOPROTO(nvnCommandBufferSetExclusiveScissors, void, NVNcommandBuffer *c, int first, int n, const int *rects) {
    if (n < 0 || first < 0 || first + n > 16) { LOG("[RED] SetExclusiveScissors first=%d n=%d", first, n); return; }
    p_excl_scissor *r = RECN(c, OP_EXCL_SCISSOR, p_excl_scissor, r, n); r->what = 1; r->first = first; r->n = n; memcpy(r->r, rects, 16 * (size_t)n);
}
NVN_NOPROTO(nvnCommandBufferSetExclusiveScissor, void, NVNcommandBuffer *c, int x, int y, int w, int h) { const int r[4] = {x, y, w, h}; h_nvnCommandBufferSetExclusiveScissors(c, 0, 1, r); }
// Coverage modulation scales colour by the share of raster samples covered when there are more of them than colour samples (mixed samples); the replay rasterizes at the
// colour sample count, where the table changes nothing
NVN(nvnCommandBufferBindCoverageModulationTable, void, NVNcommandBuffer *c, const float *t) { (void)c; (void)t; ONCE_LOG("BindCoverageModulationTable: the host rasterizes at the colour sample count, where coverage modulation changes nothing"); }
// Shader subroutines: the translated shaders call no subroutines (the recompiler has no subroutine uniforms), so the linkage is accepted and a selection has no effect
NVN(nvnProgramSetSubroutineLinkage, NVNboolean, NVNprogram *p, int n, const NVNsubroutineLinkageMapPtr *maps) { (void)p; (void)n; (void)maps; return 1; }
NVN(nvnCommandBufferSetProgramSubroutines, void, NVNcommandBuffer *c, NVNprogram *p, NVNshaderStage st, const int first, const int n, const int *values) {
    (void)c; (void)p; (void)values; ONCE_LOG("[RED] SetProgramSubroutines (stage %d, %d from %d): the translated shaders have no subroutine uniforms, the selection is not applied", (int)st, n, first);
}
// Checkpoints: the retail driver records nothing and returns true (sdk 0x64bf60 / 0x64bf70)
NVN_NOPROTO(nvnCommandBufferInsertCheckpointStatic, NVNboolean, NVNcommandBuffer *c) { (void)c; return 1; }
NVN_NOPROTO(nvnCommandBufferInsertCheckpointDynamic, NVNboolean, NVNcommandBuffer *c) { (void)c; return 1; }

// Debug groups and markers: VK_EXT_debug_utils labels in the replay's command buffers when the host has them (a debugging tool attached or renderer debug on: vka_exec.c),
// else the records are walked past. The name is copied (the Dynamic forms' string may be gone by submit; the Static ones' is kept by the game, copied all the same)
static void debug_label(NVNcommandBuffer *c, int what, uint32_t domain, const char *name) {
    size_t n = what == LABEL_POP || !name ? 0 : strnlen(name, 255);
    p_debug_label *r = rec(c, OP_DEBUG_LABEL, offsetof(p_debug_label, name) + n + 1); r->what = what; r->domain = domain;
    if (n) memcpy(r->name, name, n);
}
NVN(nvnCommandBufferPushDebugGroupStatic, void, NVNcommandBuffer *c, uint32_t d, const char *s) { debug_label(c, LABEL_PUSH, d, s); }
NVN(nvnCommandBufferPushDebugGroupDynamic, void, NVNcommandBuffer *c, uint32_t d, const char *s) { debug_label(c, LABEL_PUSH, d, s); }
NVN(nvnCommandBufferPushDebugGroup, void, NVNcommandBuffer *c, uint32_t d, const char *s) { debug_label(c, LABEL_PUSH, d, s); }
NVN_NOPROTO(nvnCommandBufferPushDebugGroupGraphics, void, NVNcommandBuffer *c, uint32_t d, const char *s) { debug_label(c, LABEL_PUSH, d, s); }
NVN_NOPROTO(nvnCommandBufferPushDebugGroupCompute, void, NVNcommandBuffer *c, uint32_t d, const char *s) { debug_label(c, LABEL_PUSH, d, s); }
NVN(nvnCommandBufferPopDebugGroup, void, NVNcommandBuffer *c) { debug_label(c, LABEL_POP, 0, NULL); }
NVN(nvnCommandBufferPopDebugGroupId, void, NVNcommandBuffer *c, uint32_t d) { debug_label(c, LABEL_POP, d, NULL); }
NVN_NOPROTO(nvnCommandBufferPopDebugGroupGraphics, void, NVNcommandBuffer *c, uint32_t d) { debug_label(c, LABEL_POP, d, NULL); }
NVN_NOPROTO(nvnCommandBufferPopDebugGroupCompute, void, NVNcommandBuffer *c, uint32_t d) { debug_label(c, LABEL_POP, d, NULL); }
NVN(nvnCommandBufferInsertDebugMarkerStatic, void, NVNcommandBuffer *c, uint32_t d, const char *s) { debug_label(c, LABEL_MARKER, d, s); }
NVN(nvnCommandBufferInsertDebugMarkerDynamic, void, NVNcommandBuffer *c, uint32_t d, const char *s) { debug_label(c, LABEL_MARKER, d, s); }
NVN(nvnCommandBufferInsertDebugMarker, void, NVNcommandBuffer *c, const char *s) { debug_label(c, LABEL_MARKER, 0, s); }
NVN_NOPROTO(nvnCommandBufferInsertDebugMarkerGraphics, void, NVNcommandBuffer *c, uint32_t d, const char *s) { debug_label(c, LABEL_MARKER, d, s); }
NVN_NOPROTO(nvnCommandBufferInsertDebugMarkerCompute, void, NVNcommandBuffer *c, uint32_t d, const char *s) { debug_label(c, LABEL_MARKER, d, s); }

// Variable-rate shading (VK_KHR_fragment_shading_rate: vka_draw.c turns the palette indices of the bound image into a shading-rate attachment of the passes that draw with it).
// BindShadingRateImage(texture, view; NULL unbinds). ShadingRateImagePalette(viewport, count, rates): the driver packs 16 4-bit entries and fills those past count (or all of
// them for NULL rates) with 5, one invocation a pixel. The palette mode (1 or another value, one method bit) is not known; logged once
NVN_NOPROTO(nvnCommandBufferBindShadingRateImage, void, NVNcommandBuffer *c, const NVNtexture *t, const NVNtextureView *v) {
    p_vrs_image *o = REC_SNAP(c, OP_VRS_IMAGE, p_vrs_image, 1); o->tex = tex_snap(o, SNAP_OFF(p_vrs_image), 0, t, nvn_caller()); copy_view(&o->view, &o->has_view, v);
}
NVN_NOPROTO(nvnCommandBufferShadingRateImagePalette, void, NVNcommandBuffer *c, int viewport, int n, const int *rates) {
    p_vrs_palette *r = REC(c, OP_VRS_PALETTE, p_vrs_palette); r->viewport = viewport & 15;
    for (int i = 0; i < 16; i++) r->rate[i] = rates && i < n ? (uint8_t)(rates[i] & 15) : 5;
}
NVN_NOPROTO(nvnCommandBufferSetShadingRatePaletteMode, void, NVNcommandBuffer *c, int mode) { (void)c; ONCE_LOG("SetShadingRatePaletteMode %d: the host maps the palette the same in every mode", mode); }

// Deferred binds and push constants: the GPU reads the bindings at src when it runs the command; the replay reads them there when it walks the record (vka_draw.c)
static void bind_defer(NVNcommandBuffer *c, int kind, NVNshaderStage st, int first, int n, NVNbufferAddress src) {
    if (n < 0 || n > 32 || first < 0) { LOG("[RED] deferred bind of %d from %d", n, first); return; }
    *REC(c, OP_BIND_DEFER, p_bind_defer) = (p_bind_defer){kind, (int32_t)st, first, n, src, CB(c)->tex_base, CB(c)->samp_base};
}
NVN_NOPROTO(nvnCommandBufferBindUniformBuffersDeferred, void, NVNcommandBuffer *c, NVNshaderStage st, int first, int n, NVNbufferAddress src) { bind_defer(c, DEFER_UBO, st, first, n, src); }
NVN_NOPROTO(nvnCommandBufferBindStorageBuffersDeferred, void, NVNcommandBuffer *c, NVNshaderStage st, int first, int n, NVNbufferAddress src) { bind_defer(c, DEFER_SSBO, st, first, n, src); }
NVN_NOPROTO(nvnCommandBufferBindImagesDeferred, void, NVNcommandBuffer *c, NVNshaderStage st, int first, int n, NVNbufferAddress src) { bind_defer(c, DEFER_IMG, st, first, n, src); }
NVN_NOPROTO(nvnCommandBufferBindSeparateTexturesDeferred, void, NVNcommandBuffer *c, NVNshaderStage st, int first, int n, NVNbufferAddress src) { bind_defer(c, DEFER_TEX, st, NVN_SEPTEX_SLOT0 + first, n, src); }
NVN_NOPROTO(nvnCommandBufferBindSeparateSamplersDeferred, void, NVNcommandBuffer *c, NVNshaderStage st, int first, int n, NVNbufferAddress src) { bind_defer(c, DEFER_SAMP, st, first, n, src); }
static void push_defer(NVNcommandBuffer *c, int compute, uint64_t w, uint64_t off, uint64_t n, NVNbufferAddress src) {
    if (w) ONCE_LOG("[RED] deferred push constants with the word before the offset %#llx: its meaning is not known", (unsigned long long)w);
    if (off + n > DMSH_PUSH_SIZE || !src) { static int k; if (k++ < 8) LOG("[RED] deferred push constants at %#llx + %#llx: beyond the host's %#x-byte block, dropped", (unsigned long long)off, (unsigned long long)n, DMSH_PUSH_SIZE); return; }
    *REC(c, OP_PUSH_DEFER, p_push_defer) = (p_push_defer){(uint16_t)compute, (uint16_t)off, (uint32_t)n, src};
}
NVN_NOPROTO(nvnCommandBufferPushGraphicsConstantsDeferred, void, NVNcommandBuffer *c, uint64_t w, uint64_t off, uint64_t n, NVNbufferAddress src) { push_defer(c, 0, w, off, n, src); }
NVN_NOPROTO(nvnCommandBufferPushComputeConstantsDeferred, void, NVNcommandBuffer *c, uint64_t w, uint64_t off, uint64_t n, NVNbufferAddress src) { push_defer(c, 1, w, off, n, src); }

// ---------- Lichii: the teardown ----------
// nvn_direct_destroy, after the replay's (vka_destroy): the virtual pools' address ranges (mappings of guest memory), the host segments of command memory, the tables
void nvn_hle_destroy(void) {
    for (int i = 0; i < g_nvp; i++) {
        struct vpool *v = g_vp[i];
#if defined(_WIN32)
        nvn_direct_alias_release(v->base);
#else
        munmap(v->base, v->len);
#endif
        for (uint64_t l = 0; l < (v->size + (2u << 20) - 1) >> 21; l++) free(v->pt[l]);
        free(v->pt); free(v);
    }
    free(g_vp);
    for (int i = 0; i < g_nhostseg; i++) free(g_hostseg[i].mem);
    free(g_hostseg);
    for (int i = 0; i < g_npc; i++) { free(g_pc[i].bb); free(g_pc[i].tb); free(g_pc[i].mb); }
    free(g_pc); free(g_pl); free(g_ft); free(g_fl);
}
