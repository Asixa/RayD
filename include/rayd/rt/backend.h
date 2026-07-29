// Copyright Xingyu Chen.
// Defines shared rt support for backend.

#pragma once

#include <cstdint>

namespace rayd::shared::rt {

// Backend-neutral trace-backend descriptors (RAY_TRACING_BACKEND_ARCHITECTURE.md
// §5, §12). These are host-safe enums/POD only: they name backends and report
// capabilities, but include no CUDA/OptiX/Embree headers and use no device
// qualifiers, so a third backend can include this header cleanly.

// Which triangle traversal backend a scene resolves to. `Auto` is a request that
// is resolved at construction to a concrete kind; `None` means no triangle trace
// backend was built (edge-only scenes, or a machine without the OptiX driver).
enum class TraceBackendKind : std::uint8_t { Auto, Optix, Cuda, Embree, None };

// The integration axis (§4.2): whether a backend folds into a Dr.Jit symbolic
// megakernel or runs as an eager native batch dispatch.
enum class IntegrationMode : std::uint8_t { JitSymbolic, EagerNative };

// Static capability report for a trace backend. All flags default to false so a
// backend only advertises what it actually supports.
struct TraceCapabilities {
    bool closest_hit = false;
    bool any_hit = false;
    bool first_blocker = false;
    bool ignore_primitives = false;
    bool instancing = false;
    bool refit = false;
    bool compaction = false;
    bool device_callable = false; ///< Traverser can inline the trace on-device.
    bool jit_symbolic = false;    ///< Can fold into a Dr.Jit megakernel (axis two).
    bool fused_multipath = false;
    bool cpu = false;
};

} // namespace rayd::shared::rt
