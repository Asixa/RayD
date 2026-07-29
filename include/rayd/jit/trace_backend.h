// Copyright Xingyu Chen.
// Declares the Dr.Jit trace backend API.

#pragma once

#include <rayd/rt/backend.h>

namespace rayd {

// Bring the backend-neutral descriptors into the rayd namespace so the Dr.Jit
// frontend (Scene, bindings) can refer to them without the shared::rt qualifier.
using shared::rt::IntegrationMode;
using shared::rt::TraceBackendKind;
using shared::rt::TraceCapabilities;

/// \brief Host-side lifecycle interface for a triangle trace backend.
///
/// This interface is deliberately narrow: it covers backend lifecycle and
/// capability introspection only. Per-ray batch trace methods are an eager-axis
/// concern that arrives in a later phase (RAY_TRACING_BACKEND_ARCHITECTURE.md §5)
/// and are intentionally absent here, so no virtual call can land in a per-ray
/// hot loop. Concrete backends (today only OptixTraceBackend) expose their own
/// typed build/sync/trace entry points.
class TraceBackend {
public:
    virtual ~TraceBackend() = default;

    /// Concrete backend kind (never Auto or None for a live backend).
    virtual TraceBackendKind kind() const = 0;
    /// Static capability report for this backend.
    virtual TraceCapabilities capabilities() const = 0;
    /// True once the backend's acceleration structures are built and refit-clean.
    virtual bool is_ready() const = 0;
};

} // namespace rayd
