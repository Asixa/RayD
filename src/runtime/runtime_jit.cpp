// Copyright Xingyu Chen.
// Implements runtime support for runtime Dr.Jit.

#include <drjit-core/optix.h>
#define OPTIX_STUBS_IMPL
#include <rayd/jit/optix.h>
#undef OPTIX_STUBS_IMPL

#include <rayd/jit/native_launch_audit.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#  define NOMINMAX
#  include <windows.h>
#  include <winver.h>
#  include <cfgmgr32.h>
#elif defined(__linux__) || defined(__APPLE__)
#  include <dlfcn.h>
#endif

namespace rayd {

namespace {

using OptixQueryFunctionTableFn = OptixResult (*)(int,
                                                  unsigned int,
                                                  OptixQueryFunctionTableOptions *,
                                                  const void **,
                                                  void *,
                                                  size_t);

std::string format_optix_version(int version) {
    std::ostringstream oss;
    oss << (version / 10000) << '.'
        << ((version / 100) % 100) << '.'
        << (version % 100);
    return oss.str();
}

#if defined(_WIN32)
std::string narrow_utf8(const std::wstring &value) {
    if (value.empty())
        return {};

    int size = WideCharToMultiByte(CP_UTF8, 0, value.data(),
                                   static_cast<int>(value.size()),
                                   nullptr, 0, nullptr, nullptr);
    if (size <= 0)
        return {};

    std::string result(static_cast<size_t>(size), '\0');
    WideCharToMultiByte(CP_UTF8, 0, value.data(),
                        static_cast<int>(value.size()),
                        result.data(), size, nullptr, nullptr);
    return result;
}

HMODULE optix_module_handle_from_symbol() {
    HMODULE module = nullptr;
    if (optixModuleCreate != nullptr &&
        GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(reinterpret_cast<const void *>(optixModuleCreate)),
                           &module)) {
        return module;
    }

    return GetModuleHandleW(L"nvoptix.dll");
}

std::string optix_module_path(HMODULE module) {
    if (!module)
        return {};

    std::wstring path(MAX_PATH, L'\0');
    DWORD size = GetModuleFileNameW(module, path.data(), static_cast<DWORD>(path.size()));
    while (size == path.size()) {
        path.resize(path.size() * 2);
        size = GetModuleFileNameW(module, path.data(), static_cast<DWORD>(path.size()));
    }

    if (size == 0)
        return {};

    path.resize(size);
    return narrow_utf8(path);
}

std::string optix_module_version(HMODULE module) {
    if (!module)
        return {};

    std::string path_utf8 = optix_module_path(module);
    if (path_utf8.empty())
        return {};

    std::wstring path(MAX_PATH, L'\0');
    DWORD path_size = GetModuleFileNameW(module, path.data(), static_cast<DWORD>(path.size()));
    while (path_size == path.size()) {
        path.resize(path.size() * 2);
        path_size = GetModuleFileNameW(module, path.data(), static_cast<DWORD>(path.size()));
    }
    if (path_size == 0)
        return {};
    path.resize(path_size);

    DWORD unused = 0;
    DWORD info_size = GetFileVersionInfoSizeW(path.c_str(), &unused);
    if (info_size == 0)
        return {};

    std::string buffer(static_cast<size_t>(info_size), '\0');
    if (!GetFileVersionInfoW(path.c_str(), 0, info_size, buffer.data()))
        return {};

    VS_FIXEDFILEINFO *info = nullptr;
    UINT len = 0;
    if (!VerQueryValueW(buffer.data(), L"\\", reinterpret_cast<LPVOID *>(&info), &len) ||
        info == nullptr || len < sizeof(VS_FIXEDFILEINFO)) {
        return {};
    }

    std::ostringstream oss;
    oss << HIWORD(info->dwFileVersionMS) << '.'
        << LOWORD(info->dwFileVersionMS) << '.'
        << HIWORD(info->dwFileVersionLS) << '.'
        << LOWORD(info->dwFileVersionLS);
    return oss.str();
}

OptixQueryFunctionTableFn optix_query_function_table(HMODULE module) {
    if (!module)
        return nullptr;
    return reinterpret_cast<OptixQueryFunctionTableFn>(
        GetProcAddress(module, "optixQueryFunctionTable"));
}
#elif defined(__linux__) || defined(__APPLE__)
void *optix_module_handle_from_symbol() {
    Dl_info info;
    if (optixModuleCreate != nullptr &&
        dladdr(reinterpret_cast<const void *>(optixModuleCreate), &info) != 0 &&
        info.dli_fname != nullptr) {
        return dlopen(info.dli_fname, RTLD_LAZY | RTLD_LOCAL);
    }
    return dlopen("libnvoptix.so.1", RTLD_LAZY | RTLD_LOCAL);
}

std::string optix_module_path(void *module) {
    Dl_info info;
    if (optixModuleCreate != nullptr &&
        dladdr(reinterpret_cast<const void *>(optixModuleCreate), &info) != 0 &&
        info.dli_fname != nullptr) {
        return info.dli_fname;
    }
    (void) module;
    return {};
}

std::string optix_module_version(void *) {
    return {};
}

OptixQueryFunctionTableFn optix_query_function_table(void *module) {
    if (!module)
        return nullptr;
    return reinterpret_cast<OptixQueryFunctionTableFn>(
        dlsym(module, "optixQueryFunctionTable"));
}
#else
void *optix_module_handle_from_symbol() { return nullptr; }
std::string optix_module_path(void *) { return {}; }
std::string optix_module_version(void *) { return {}; }
OptixQueryFunctionTableFn optix_query_function_table(void *) { return nullptr; }
#endif

// Standalone driver-module probe used by optix_available(). Unlike
// optix_module_handle_from_symbol(), this actively *loads* the driver library
// (GetModuleHandleW does not load), so it works before any OptiX symbol resolves.
#if defined(_WIN32)
// Locate nvoptix.dll the way the OptiX SDK's optixLoadWindowsDll does: it lives
// in the NVIDIA driver store, not on the default DLL search path, so a bare
// LoadLibrary("nvoptix.dll") fails. Fall back to the graphics-driver directory
// found via the display-adapter device's OpenGLDriverName registry value.
HMODULE optix_load_windows_dll_from_driver_store(const char *dll_name) {
    static const char *kDisplayAdapterClassGuid =
        "{4d36e968-e325-11ce-bfc1-08002be10318}";
    const ULONG flags = CM_GETIDLIST_FILTER_CLASS | CM_GETIDLIST_FILTER_PRESENT;

    ULONG device_list_size = 0;
    if (CM_Get_Device_ID_List_SizeA(&device_list_size, kDisplayAdapterClassGuid, flags) !=
            CR_SUCCESS ||
        device_list_size == 0) {
        return nullptr;
    }

    std::string device_names(device_list_size, '\0');
    if (CM_Get_Device_ID_ListA(kDisplayAdapterClassGuid, device_names.data(),
                               device_list_size, flags) != CR_SUCCESS) {
        return nullptr;
    }

    for (const char *device_name = device_names.c_str(); *device_name != '\0';
         device_name += std::strlen(device_name) + 1) {
        DEVINST device_id = 0;
        if (CM_Locate_DevNodeA(&device_id, const_cast<DEVINSTID_A>(device_name),
                               CM_LOCATE_DEVNODE_NORMAL) != CR_SUCCESS) {
            continue;
        }

        HKEY reg_key = nullptr;
        if (CM_Open_DevNode_Key(device_id, KEY_QUERY_VALUE, 0, RegDisposition_OpenExisting,
                                &reg_key, CM_REGISTRY_SOFTWARE) != CR_SUCCESS) {
            continue;
        }

        DWORD value_size = 0;
        if (RegQueryValueExA(reg_key, "OpenGLDriverName", nullptr, nullptr, nullptr,
                             &value_size) != ERROR_SUCCESS) {
            RegCloseKey(reg_key);
            continue;
        }

        std::string reg_value(value_size, '\0');
        const LSTATUS status =
            RegQueryValueExA(reg_key, "OpenGLDriverName", nullptr, nullptr,
                             reinterpret_cast<LPBYTE>(reg_value.data()), &value_size);
        RegCloseKey(reg_key);
        if (status != ERROR_SUCCESS) {
            continue;
        }

        // reg_value is the full path to the OpenGL driver DLL (possibly with a
        // trailing NUL). Replace its file name with dll_name and try to load it.
        const std::string driver_dll(reg_value.c_str());
        const size_t slash = driver_dll.find_last_of('\\');
        if (slash == std::string::npos) {
            continue;
        }
        const std::string candidate = driver_dll.substr(0, slash + 1) + dll_name;
        if (HMODULE handle = LoadLibraryA(candidate.c_str())) {
            return handle;
        }
    }
    return nullptr;
}

HMODULE probe_load_optix_module() {
    // Already loaded (for example after Dr.Jit initialized OptiX): reuse it.
    if (HMODULE module = GetModuleHandleW(L"nvoptix.dll")) {
        return module;
    }
    // On the default search path (rare) or in System32.
    if (HMODULE module = LoadLibraryW(L"nvoptix.dll")) {
        return module;
    }
    return optix_load_windows_dll_from_driver_store("nvoptix.dll");
}
#elif defined(__linux__) || defined(__APPLE__)
void *probe_load_optix_module() {
    return dlopen("libnvoptix.so.1", RTLD_LAZY | RTLD_LOCAL);
}
#else
void *probe_load_optix_module() { return nullptr; }
#endif

// Kill-switch: RAYD_DISABLE_OPTIX=1/true forces optix_available() to report false,
// letting an OptiX-capable machine exercise the OptiX-less paths in tests/CI.
bool env_disables_optix() {
    const char *raw = std::getenv("RAYD_DISABLE_OPTIX");
    if (raw == nullptr)
        return false;
    std::string value(raw);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value == "1" || value == "true";
}

struct OptixDeviceContextOptionsProbe {
    void *log_callback_function = nullptr;
    void *log_callback_data = nullptr;
    int log_callback_level = 0;
    unsigned int validation_mode = 0;
};

using OptixDeviceContextCreateFn = OptixResult (*)(
    void *, const OptixDeviceContextOptionsProbe *, OptixDeviceContext *);
using OptixDeviceContextDestroyFn = OptixResult (*)(OptixDeviceContext);

constexpr OptixResult kOptixErrorNotCompatible = 7400;
constexpr OptixResult kOptixErrorNotSupported = 7800;

std::mutex optix_capability_mutex;
std::map<std::pair<int, std::uintptr_t>, bool> optix_capability_cache;

} // namespace

OptixRuntimeInfo query_optix_runtime_info() {
    jit_optix_context();
    init_optix_api();

    OptixRuntimeInfo info;
    info.module_create_available = optixModuleCreate != nullptr;
    info.device_context_get_property_available = optixDeviceContextGetProperty != nullptr;

    auto module = optix_module_handle_from_symbol();
    info.module_path = optix_module_path(module);
    info.module_version = optix_module_version(module);

    OptixQueryFunctionTableFn query_fn = optix_query_function_table(module);
    info.query_function_table_available = query_fn != nullptr;
    if (query_fn != nullptr) {
        info.abi_probe_result = query_fn(RAYD_OPTIX_TARGET_ABI,
                                         0,
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         0);
        info.target_abi_supported = info.abi_probe_result != 7801;
    }

    if (optixDeviceContextGetProperty != nullptr) {
        unsigned int rtcore_version = 0;
        OptixResult rv = optixDeviceContextGetProperty(
            jit_optix_context(),
            OPTIX_DEVICE_PROPERTY_RTCORE_VERSION,
            &rtcore_version,
            sizeof(rtcore_version));
        if (rv == 0)
            info.rtcore_version = static_cast<int>(rtcore_version);
    }

#if defined(__linux__) || defined(__APPLE__)
    if (module != nullptr)
        dlclose(module);
#endif

    return info;
}

bool optix_available() {
    if (env_disables_optix())
        return false;

    void *cuda_context = jit_cuda_context();
    if (cuda_context == nullptr) {
        throw std::runtime_error(
            "OptiX capability probe requires an active Dr.Jit CUDA context.");
    }
    const int device = jit_cuda_device();
    const auto key = std::make_pair(
        device, reinterpret_cast<std::uintptr_t>(cuda_context));

    std::lock_guard<std::mutex> lock(optix_capability_mutex);
    auto cached = optix_capability_cache.find(key);
    if (cached != optix_capability_cache.end())
        return cached->second;

    // Retain one process-lifetime driver-module reference. Repeated scene
    // construction must not accumulate dlopen reference counts on Linux.
    static auto module = probe_load_optix_module();
    if (!module) {
        optix_capability_cache.emplace(key, false);
        return false;
    }

    OptixQueryFunctionTableFn query_fn = optix_query_function_table(module);
    if (query_fn == nullptr) {
        optix_capability_cache.emplace(key, false);
        return false;
    }

    // OptiX 8.1 ABI-93 contains exactly 50 pointer-sized entries. Query a
    // private table so capability discovery does not initialize Dr.Jit's
    // default OptiX module/program/SBT. Entries 2 and 3 are the typed device
    // context create/destroy functions in NVIDIA's ABI-93 contract.
    std::array<void *, 50> function_table = {};
    const OptixResult abi_probe = query_fn(
        RAYD_OPTIX_TARGET_ABI,
        0,
        nullptr,
        nullptr,
        function_table.data(),
        sizeof(function_table));
    if (abi_probe == 7801) {
        optix_capability_cache.emplace(key, false);
        return false;
    }
    if (abi_probe != 0) {
        throw std::runtime_error(
            "OptiX ABI capability probe failed with error code " +
            std::to_string(abi_probe) + ".");
    }

    auto create_context = reinterpret_cast<OptixDeviceContextCreateFn>(
        function_table[2]);
    auto destroy_context = reinterpret_cast<OptixDeviceContextDestroyFn>(
        function_table[3]);
    if (create_context == nullptr || destroy_context == nullptr) {
        throw std::runtime_error(
            "OptiX ABI-93 function table is missing device-context entry points.");
    }

    OptixDeviceContextOptionsProbe options = {};
    OptixDeviceContext temporary_context = nullptr;
    const OptixResult create_result =
        create_context(cuda_context, &options, &temporary_context);
    if (create_result == kOptixErrorNotSupported ||
        create_result == kOptixErrorNotCompatible) {
        optix_capability_cache.emplace(key, false);
        return false;
    }
    if (create_result != 0) {
        throw std::runtime_error(
            "OptiX device capability probe failed with error code " +
            std::to_string(create_result) + ".");
    }

    const OptixResult destroy_result = destroy_context(temporary_context);
    if (destroy_result != 0) {
        throw std::runtime_error(
            "OptiX device capability probe cleanup failed with error code " +
            std::to_string(destroy_result) + ".");
    }
    optix_capability_cache.emplace(key, true);
    return true;
}

void check_optix(OptixResult result, const char *message) {
    if (result != 0) {
        throw std::runtime_error(std::string("OptiX error in ") + message);
    }
}

OptixProgramGroup make_raygen_group(OptixDeviceContext context,
                                    OptixModule module,
                                    const char *entry_name) {
    OptixProgramGroupOptions pg_options = {};
    OptixProgramGroupDesc desc = {};
    desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    desc.raygen.module = module;
    desc.raygen.entryFunctionName = entry_name;

    char log[2048];
    size_t log_size = sizeof(log);
    OptixProgramGroup group = nullptr;
    check_optix(optixProgramGroupCreate(context, &desc, 1, &pg_options, log, &log_size, &group),
                "optixProgramGroupCreate(raygen)");
    return group;
}

OptixProgramGroup make_miss_group(OptixDeviceContext context,
                                  OptixModule module,
                                  const char *entry_name) {
    OptixProgramGroupOptions pg_options = {};
    OptixProgramGroupDesc desc = {};
    desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    desc.miss.module = module;
    desc.miss.entryFunctionName = entry_name;

    char log[2048];
    size_t log_size = sizeof(log);
    OptixProgramGroup group = nullptr;
    check_optix(optixProgramGroupCreate(context, &desc, 1, &pg_options, log, &log_size, &group),
                "optixProgramGroupCreate(miss)");
    return group;
}

OptixProgramGroup make_hitgroup(OptixDeviceContext context,
                                OptixModule module,
                                const char *closesthit,
                                const char *anyhit,
                                const char *intersection) {
    OptixProgramGroupOptions pg_options = {};
    OptixProgramGroupDesc desc = {};
    desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    desc.hitgroup.moduleCH = closesthit != nullptr ? module : nullptr;
    desc.hitgroup.entryFunctionNameCH = closesthit;
    desc.hitgroup.moduleAH = anyhit != nullptr ? module : nullptr;
    desc.hitgroup.entryFunctionNameAH = anyhit;
    desc.hitgroup.moduleIS = intersection != nullptr ? module : nullptr;
    desc.hitgroup.entryFunctionNameIS = intersection;

    char log[2048];
    size_t log_size = sizeof(log);
    OptixProgramGroup group = nullptr;
    check_optix(optixProgramGroupCreate(context, &desc, 1, &pg_options, log, &log_size, &group),
                "optixProgramGroupCreate(hitgroup)");
    return group;
}

void *make_sbt_record(OptixProgramGroup group) {
    EmptySbtRecord record = {};
    check_optix(optixSbtRecordPackHeader(group, &record), "optixSbtRecordPackHeader");

    void *device_record = jit_malloc(AllocType::Device, sizeof(EmptySbtRecord));
    audit_jit_memcpy();
    jit_memcpy(JitBackend::CUDA, device_record, &record, sizeof(EmptySbtRecord));
    return device_record;
}

} // namespace rayd

void init_optix_api() {
    jit_optix_context(); // Ensure OptiX is initialized

    #define L(name) name = (decltype(name)) jit_optix_lookup(#name);

    L(optixAccelComputeMemoryUsage);
    L(optixAccelBuild);
    L(optixAccelCompact);
    L(optixModuleCreate);
    L(optixDeviceContextGetProperty);
    L(optixModuleDestroy)
    L(optixProgramGroupCreate);
    L(optixProgramGroupDestroy)
    L(optixPipelineCreate);
    L(optixPipelineDestroy);
    L(optixPipelineSetStackSize);
    L(optixSbtRecordPackHeader);
    L(optixLaunch);

    #undef L
}

// Consolidated native launch audit implementation.
#include <rayd/jit/native_launch_audit.h>

#include <algorithm>
#include <cstdlib>
#include <mutex>

namespace rayd {

namespace {

std::mutex &audit_mutex() {
    static std::mutex mutex;
    return mutex;
}

NativeLaunchAuditSnapshot &audit_snapshot_storage() {
    static NativeLaunchAuditSnapshot snapshot;
    return snapshot;
}

thread_local NativeLaunchStage current_stage = NativeLaunchStage::Unknown;

NativeLaunchStageStats &stage_stats(NativeLaunchAuditSnapshot &snapshot,
                                    NativeLaunchStage stage) {
    switch (stage) {
    case NativeLaunchStage::Build:
        return snapshot.build;
    case NativeLaunchStage::Sync:
        return snapshot.sync;
    case NativeLaunchStage::Intersect:
        return snapshot.intersect;
    case NativeLaunchStage::TraceReflections:
        return snapshot.trace_reflections;
    case NativeLaunchStage::AccumulateReflections:
        return snapshot.accumulate_reflections;
    case NativeLaunchStage::AccumDfr:
        return snapshot.accum_dfr;
    case NativeLaunchStage::SurfelTrace:
        return snapshot.surfel_trace;
    case NativeLaunchStage::Unknown:
    default:
        return snapshot.unknown;
    }
}

void clear_stage_stats(NativeLaunchStageStats &stats) {
    stats = NativeLaunchStageStats();
}

void update_kernel_stat(NativeLaunchStageStats &stats,
                        const char *label,
                        uint64_t threads,
                        uint64_t items) {
    const std::string key = label != nullptr ? std::string(label) : std::string("unnamed");
    auto it = std::find_if(stats.kernels.begin(),
                           stats.kernels.end(),
                           [&key](const NativeKernelLaunchStat &entry) {
                               return entry.label == key;
                           });
    if (it == stats.kernels.end()) {
        stats.kernels.push_back(NativeKernelLaunchStat { key, 1, threads, threads, items, items });
        return;
    }

    it->launches += 1;
    it->total_threads += threads;
    it->max_threads = std::max(it->max_threads, threads);
    it->total_items += items;
    it->max_items = std::max(it->max_items, items);
}

template <typename Member>
void increment_counter(Member NativeLaunchStageStats::*member) {
    std::lock_guard<std::mutex> guard(audit_mutex());
    NativeLaunchStageStats &stats = stage_stats(audit_snapshot_storage(), current_stage);
    stats.*member += 1;
}

} // namespace

ScopedNativeLaunchStage::ScopedNativeLaunchStage(NativeLaunchStage stage)
    : previous_(current_stage) {
    current_stage = stage;
}

ScopedNativeLaunchStage::~ScopedNativeLaunchStage() {
    current_stage = previous_;
}

void native_launch_audit_clear() {
    std::lock_guard<std::mutex> guard(audit_mutex());
    NativeLaunchAuditSnapshot &snapshot = audit_snapshot_storage();
    clear_stage_stats(snapshot.unknown);
    clear_stage_stats(snapshot.build);
    clear_stage_stats(snapshot.sync);
    clear_stage_stats(snapshot.intersect);
    clear_stage_stats(snapshot.trace_reflections);
    clear_stage_stats(snapshot.accumulate_reflections);
    clear_stage_stats(snapshot.accum_dfr);
    clear_stage_stats(snapshot.surfel_trace);
}

NativeLaunchAuditSnapshot native_launch_audit_snapshot() {
    std::lock_guard<std::mutex> guard(audit_mutex());
    return audit_snapshot_storage();
}

void audit_cuda_kernel_launch(const char *label,
                              uint32_t grid_x,
                              uint32_t grid_y,
                              uint32_t grid_z,
                              uint32_t block_x,
                              uint32_t block_y,
                              uint32_t block_z,
                              uint64_t items) {
    const uint64_t threads =
        static_cast<uint64_t>(grid_x) * static_cast<uint64_t>(grid_y) * static_cast<uint64_t>(grid_z) *
        static_cast<uint64_t>(block_x) * static_cast<uint64_t>(block_y) * static_cast<uint64_t>(block_z);

    std::lock_guard<std::mutex> guard(audit_mutex());
    NativeLaunchStageStats &stats = stage_stats(audit_snapshot_storage(), current_stage);
    stats.cuda_kernel_launches += 1;
    stats.cuda_kernel_total_threads += threads;
    update_kernel_stat(stats, label, threads, items);
}

void audit_cuda_memcpy() {
    increment_counter(&NativeLaunchStageStats::cuda_memcpy);
}

void audit_cuda_memcpy_async() {
    increment_counter(&NativeLaunchStageStats::cuda_memcpy_async);
}

void audit_cuda_memset_async() {
    increment_counter(&NativeLaunchStageStats::cuda_memset_async);
}

void audit_cuda_stream_synchronize() {
    increment_counter(&NativeLaunchStageStats::cuda_stream_synchronize);
}

void audit_cuda_event_record() {
    increment_counter(&NativeLaunchStageStats::cuda_event_record);
}

void audit_cuda_stream_wait_event() {
    increment_counter(&NativeLaunchStageStats::cuda_stream_wait_event);
}

void audit_cub_reduce() {
    increment_counter(&NativeLaunchStageStats::cub_reduce);
}

void audit_cub_sort() {
    increment_counter(&NativeLaunchStageStats::cub_sort);
}

void audit_cub_scan() {
    increment_counter(&NativeLaunchStageStats::cub_scan);
}

void audit_jit_memcpy() {
    increment_counter(&NativeLaunchStageStats::jit_memcpy);
}

void audit_jit_memcpy_async() {
    increment_counter(&NativeLaunchStageStats::jit_memcpy_async);
}

void audit_optix_accel_build() {
    increment_counter(&NativeLaunchStageStats::optix_accel_build);
}

void audit_optix_accel_compact() {
    increment_counter(&NativeLaunchStageStats::optix_accel_compact);
}

void audit_optix_launch() {
    increment_counter(&NativeLaunchStageStats::optix_launch);
}

void audit_optix_launch_duration_ms(double elapsed_ms) {
    std::lock_guard<std::mutex> guard(audit_mutex());
    NativeLaunchStageStats &stats = stage_stats(audit_snapshot_storage(), current_stage);
    stats.optix_launch_time_ms += elapsed_ms;
    if (stats.optix_launch_time_min_ms == 0.0 ||
        elapsed_ms < stats.optix_launch_time_min_ms) {
        stats.optix_launch_time_min_ms = elapsed_ms;
    }
    stats.optix_launch_time_max_ms = std::max(stats.optix_launch_time_max_ms, elapsed_ms);
}

bool native_launch_audit_timing_enabled() {
    static const bool enabled = []() {
        const char *value = std::getenv("RAYD_NATIVE_LAUNCH_AUDIT_TIMING");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

} // namespace rayd

// Consolidated multipath pipeline manager.
#include <src/runtime/optix_pipelines_jit.h>

#include <algorithm>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

#include <cuda_runtime_api.h>

#include <rayd/jit/native_launch_audit.h>
#include <rayd/rt/optix_pipeline_contracts.h>

#include <reflection_trace_ptx.h>
#include <reflection_epc_ptx.h>
#include <reflection_accumulation_ptx.h>
#include <diffraction_accumulation_ptx.h>
#include <diffraction_paths_ptx.h>
#include <segment_visibility_ptx.h>
#include <src/reflection/trace_params_jit.h>
#include <src/reflection/epc_params_jit.h>
#include <src/reflection/accumulation_params_jit.h>
#include <src/diffraction/accumulation_params_jit.h>
#include <src/diffraction/paths_params_jit.h>
#include <src/visibility/segment_params_jit.h>

namespace rayd {

#ifndef RAYD_OPTIX_MODULE_OPT_LEVEL
#  define RAYD_OPTIX_MODULE_OPT_LEVEL OPTIX_COMPILE_OPTIMIZATION_LEVEL_3
#endif

#ifndef RAYD_MULTIPATH_OPTIX_MODULE_OPT_LEVEL
#  define RAYD_MULTIPATH_OPTIX_MODULE_OPT_LEVEL RAYD_OPTIX_MODULE_OPT_LEVEL
#endif

#ifndef RAYD_MULTIPATH_OPTIX_EXCEPTION_FLAGS
#  define RAYD_MULTIPATH_OPTIX_EXCEPTION_FLAGS RAYD_OPTIX_EXCEPTION_FLAGS
#endif

namespace {

using PipelineCacheKey = std::tuple<
    OptixDeviceContext,
    const char *,
    size_t,
    std::string,
    std::string,
    std::string,
    std::string,
    int,
    int,
    size_t>;

std::string pipeline_entry_key(const char *entry) {
    return entry != nullptr ? std::string(entry) : std::string();
}

std::string pipeline_raygen_entries_key(const std::vector<const char *> &entries) {
    std::string key;
    for (const char *entry : entries) {
        if (!key.empty()) {
            key.push_back('\n');
        }
        key += pipeline_entry_key(entry);
    }
    return key;
}

std::mutex &pipeline_cache_mutex() {
    static std::mutex *mutex = new std::mutex();
    return *mutex;
}

std::map<PipelineCacheKey, std::shared_ptr<OptixLaunchPipeline>> &pipeline_cache() {
    static std::map<PipelineCacheKey, std::shared_ptr<OptixLaunchPipeline>> *cache =
        new std::map<PipelineCacheKey, std::shared_ptr<OptixLaunchPipeline>>();
    return *cache;
}

void check_cuda(cudaError_t result, const char *message) {
    require(result == cudaSuccess,
            std::string(message) + ": " + cudaGetErrorString(result));
}

int hitgroup_record_capacity(int hitgroup_record_count) {
    constexpr int kMinHitgroupRecordCapacity = 64;
    int capacity = kMinHitgroupRecordCapacity;
    while (capacity < hitgroup_record_count) {
        capacity *= 2;
    }
    return capacity;
}

} // namespace

OptixLaunchPipeline::~OptixLaunchPipeline() {
    if (pipeline_ != nullptr && optixPipelineDestroy != nullptr) {
        optixPipelineDestroy(pipeline_);
    }
    if (pg_hitgroup_ != nullptr && optixProgramGroupDestroy != nullptr) {
        optixProgramGroupDestroy(pg_hitgroup_);
    }
    if (pg_miss_ != nullptr && optixProgramGroupDestroy != nullptr) {
        optixProgramGroupDestroy(pg_miss_);
    }
    for (OptixProgramGroup pg : pg_raygens_) {
        if (pg != nullptr && optixProgramGroupDestroy != nullptr) {
            optixProgramGroupDestroy(pg);
        }
    }
    if (module_ != nullptr && optixModuleDestroy != nullptr) {
        optixModuleDestroy(module_);
    }
    if (params_buffer_ != nullptr) {
        jit_free(params_buffer_);
    }
    if (sbt_hitgroup_records_ != nullptr) {
        jit_free(sbt_hitgroup_records_);
    }
    if (sbt_miss_record_ != nullptr) {
        jit_free(sbt_miss_record_);
    }
    for (void *record : sbt_raygen_records_) {
        if (record != nullptr) {
            jit_free(record);
        }
    }
}

/// Compile the module, create program groups, link the pipeline, and build the SBT and
/// params buffer from \p config. The shared build sequence for all four multipath pipelines.
void OptixLaunchPipeline::build(OptixDeviceContext context,
                                int hitgroup_record_count,
                                const OptixPipelineConfig &config) {
    require(context != nullptr, "OptixLaunchPipeline::build(): invalid OptiX context.");
    require(hitgroup_record_count > 0,
            "OptixLaunchPipeline::build(): hitgroup_record_count must be positive.");
    require(!config.raygen_entries.empty(),
            "OptixLaunchPipeline::build(): config requires at least one raygen entry.");
    init_optix_api();

    OptixModuleCompileOptions module_options = {};
    module_options.maxRegisterCount = 0;
    module_options.optLevel = RAYD_MULTIPATH_OPTIX_MODULE_OPT_LEVEL;
    module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    OptixPipelineCompileOptions pipeline_options = {};
    pipeline_options.usesMotionBlur = 0;
    pipeline_options.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipeline_options.numPayloadValues = config.num_payload_values;
    pipeline_options.numAttributeValues = shared::optix::TriangleAttributeCount;
    pipeline_options.exceptionFlags = RAYD_MULTIPATH_OPTIX_EXCEPTION_FLAGS;
    pipeline_options.pipelineLaunchParamsVariableName = "params";
    pipeline_options.usesPrimitiveTypeFlags =
        static_cast<unsigned>(OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);
    pipeline_options.allowOpacityMicromaps = 0;

    char log[2048];
    size_t log_size = sizeof(log);
    check_optix(optixModuleCreate(context,
                                  &module_options,
                                  &pipeline_options,
                                  config.ptx,
                                  config.ptx_size,
                                  log,
                                  &log_size,
                                  &module_),
                "optixModuleCreate(multipath)");

    for (const char *entry : config.raygen_entries) {
        pg_raygens_.push_back(make_raygen_group(context, module_, entry));
    }
    pg_miss_ = make_miss_group(context, module_, config.miss_entry);
    pg_hitgroup_ = make_hitgroup(context, module_, config.closesthit_entry,
                                 config.anyhit_entry, nullptr);

    std::vector<OptixProgramGroup> groups = pg_raygens_;
    groups.push_back(pg_miss_);
    groups.push_back(pg_hitgroup_);

    OptixPipelineLinkOptions link_options = {};
    link_options.maxTraceDepth = 1;
    link_options.maxContinuationCallableDepth = 0;
    link_options.maxDirectCallableDepthFromState = 0;
    link_options.maxDirectCallableDepthFromTraversal = 0;
    link_options.maxTraversableGraphDepth = 2;

    log_size = sizeof(log);
    check_optix(optixPipelineCreate(context,
                                    &pipeline_options,
                                    &link_options,
                                    groups.data(),
                                    static_cast<unsigned int>(groups.size()),
                                    log,
                                    &log_size,
                                    &pipeline_),
                "optixPipelineCreate(multipath)");

    check_optix(optixPipelineSetStackSize(pipeline_, 0, 0, 4096, 2),
                "optixPipelineSetStackSize(multipath)");

    for (OptixProgramGroup pg : pg_raygens_) {
        sbt_raygen_records_.push_back(make_sbt_record(pg));
    }
    sbt_miss_record_ = make_sbt_record(pg_miss_);

    std::vector<EmptySbtRecord> hitgroup_records(static_cast<size_t>(hitgroup_record_count));
    for (EmptySbtRecord &record : hitgroup_records) {
        check_optix(optixSbtRecordPackHeader(pg_hitgroup_, &record),
                    "optixSbtRecordPackHeader(hitgroup)");
    }
    sbt_hitgroup_records_ = jit_malloc(AllocType::Device,
                                       sizeof(EmptySbtRecord) * hitgroup_records.size());
    audit_jit_memcpy();
    jit_memcpy(JitBackend::CUDA,
               sbt_hitgroup_records_,
               hitgroup_records.data(),
               sizeof(EmptySbtRecord) * hitgroup_records.size());

    for (OptixProgramGroup &pg : pg_raygens_) {
        if (pg != nullptr) {
            check_optix(optixProgramGroupDestroy(pg),
                        "optixProgramGroupDestroy(raygen)");
            pg = nullptr;
        }
    }
    if (pg_hitgroup_ != nullptr) {
        check_optix(optixProgramGroupDestroy(pg_hitgroup_),
                    "optixProgramGroupDestroy(hitgroup)");
        pg_hitgroup_ = nullptr;
    }
    if (pg_miss_ != nullptr) {
        check_optix(optixProgramGroupDestroy(pg_miss_),
                    "optixProgramGroupDestroy(miss)");
        pg_miss_ = nullptr;
    }
    if (module_ != nullptr) {
        check_optix(optixModuleDestroy(module_), "optixModuleDestroy(multipath)");
        module_ = nullptr;
    }

    params_size_ = config.params_size;
    params_buffer_size_ = std::max<size_t>(params_size_, 1024);
    params_buffer_ = jit_malloc(AllocType::Device, params_buffer_size_);
    hitgroup_record_count_ = hitgroup_record_count;
    device_ = jit_cuda_device();
    ready_ = true;
}

std::shared_ptr<OptixLaunchPipeline> shared_optix_launch_pipeline(
    OptixDeviceContext context,
    int hitgroup_record_count,
    const OptixPipelineConfig &config) {
    int hitgroup_capacity = hitgroup_record_capacity(hitgroup_record_count);
    PipelineCacheKey key{
        context,
        config.ptx,
        config.ptx_size,
        pipeline_raygen_entries_key(config.raygen_entries),
        pipeline_entry_key(config.miss_entry),
        pipeline_entry_key(config.closesthit_entry),
        pipeline_entry_key(config.anyhit_entry),
        hitgroup_capacity,
        config.num_payload_values,
        config.params_size,
    };

    std::lock_guard<std::mutex> guard(pipeline_cache_mutex());
    auto &cache = pipeline_cache();
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }

    auto pipeline = std::make_shared<OptixLaunchPipeline>();
    pipeline->build(context, hitgroup_capacity, config);
    cache[key] = pipeline;
    return pipeline;
}

/// Upload \p params and launch the pipeline with the \p raygen_index'th raygen entry over n_rays threads.
void OptixLaunchPipeline::launch_impl(int raygen_index,
                                       const void *params,
                                       size_t actual_params_size,
                                       unsigned int n_rays) const {
    require(ready_, "OptixLaunchPipeline::launch(): pipeline is not ready.");
    // The OptiX module, SBT records, and params buffer were allocated on the
    // build-time Dr.Jit device, while the launch below uses the stream of
    // whichever device is current. Reject the mismatch instead of corrupting.
    const int current_device = jit_cuda_device();
    if (current_device != device_) {
        throw std::runtime_error(
            "OptixLaunchPipeline::launch(): pipeline was built on Dr.Jit CUDA "
            "device " + std::to_string(device_) + " but the current Dr.Jit CUDA "
            "device is " + std::to_string(current_device) +
            ". Multipath pipelines are bound to their build device; call "
            "rayd.drjit.set_device(" + std::to_string(device_) +
            ") before launching, or rebuild the scene on the current device.");
    }
    require(raygen_index >= 0 &&
                raygen_index < static_cast<int>(sbt_raygen_records_.size()),
            "OptixLaunchPipeline::launch(): raygen index out of range.");

    const size_t launch_params_size = std::max(params_size_, actual_params_size);
    require(launch_params_size <= params_buffer_size_,
            "OptixLaunchPipeline::launch(): params buffer is too small.");

    audit_jit_memcpy_async();
    jit_memcpy_async(JitBackend::CUDA, params_buffer_, params, launch_params_size);

    OptixShaderBindingTable sbt = {};
    sbt.raygenRecord = reinterpret_cast<CUdeviceptr>(sbt_raygen_records_[raygen_index]);
    sbt.missRecordBase = reinterpret_cast<CUdeviceptr>(sbt_miss_record_);
    sbt.missRecordStrideInBytes = sizeof(EmptySbtRecord);
    sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(sbt_hitgroup_records_);
    sbt.hitgroupRecordStrideInBytes = sizeof(EmptySbtRecord);
    sbt.hitgroupRecordCount = static_cast<unsigned int>(hitgroup_record_count_);

    CUstream jit_stream = jit_cuda_stream();
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(jit_stream);
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    const bool time_optix_launch = native_launch_audit_timing_enabled();
    if (time_optix_launch) {
        check_cuda(cudaEventCreateWithFlags(&start_event, cudaEventDefault),
                   "cudaEventCreateWithFlags(start)");
        check_cuda(cudaEventCreateWithFlags(&stop_event, cudaEventDefault),
                   "cudaEventCreateWithFlags(stop)");
        audit_cuda_event_record();
        check_cuda(cudaEventRecord(start_event, cuda_stream), "cudaEventRecord(start)");
    }

    audit_optix_launch();
    check_optix(optixLaunch(pipeline_,
                            jit_stream,
                            reinterpret_cast<CUdeviceptr>(params_buffer_),
                            launch_params_size,
                            &sbt,
                            n_rays,
                            1,
                            1),
                "optixLaunch(multipath)");

    if (time_optix_launch) {
        audit_cuda_event_record();
        check_cuda(cudaEventRecord(stop_event, cuda_stream), "cudaEventRecord(stop)");
        check_cuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize(stop)");
        float elapsed_ms = 0.0f;
        check_cuda(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event),
                   "cudaEventElapsedTime(optixLaunch)");
        audit_optix_launch_duration_ms(static_cast<double>(elapsed_ms));
        check_cuda(cudaEventDestroy(start_event), "cudaEventDestroy(start)");
        check_cuda(cudaEventDestroy(stop_event), "cudaEventDestroy(stop)");
    }
}

OptixPipelineConfig reflection_trace_pipeline_config() {
    OptixPipelineConfig config;
    config.ptx = reflection_trace_ptx;
    config.ptx_size = reflection_trace_ptx_size;
    config.raygen_entries = {"__raygen__reflection_trace"};
    config.miss_entry = "__miss__reflection";
    config.closesthit_entry = "__closesthit__reflection";
    config.num_payload_values = shared::optix::TriangleHitPayloadCount;
    config.params_size = sizeof(ReflectionTraceParams);
    return config;
}

OptixPipelineConfig reflection_epc_pipeline_config() {
    OptixPipelineConfig config;
    config.ptx = reflection_epc_ptx;
    config.ptx_size = reflection_epc_ptx_size;
    config.raygen_entries = {"__raygen__reflection_epc"};
    config.miss_entry = "__miss__reflection_epc";
    config.closesthit_entry = "__closesthit__reflection_epc";
    config.anyhit_entry = "__anyhit__reflection_epc";
    config.num_payload_values = shared::optix::TriangleHitPayloadCount;
    config.params_size = sizeof(ReflEpcParams);
    return config;
}

OptixPipelineConfig reflection_epc_direct_pipeline_config() {
    OptixPipelineConfig config;
    config.ptx = reflection_epc_ptx;
    config.ptx_size = reflection_epc_ptx_size;
    config.raygen_entries = {"__raygen__reflection_epc_direct"};
    config.miss_entry = "__miss__reflection_epc";
    config.closesthit_entry = "__closesthit__reflection_epc";
    config.anyhit_entry = "__anyhit__reflection_epc";
    config.num_payload_values = shared::optix::TriangleHitPayloadCount;
    config.params_size = sizeof(ReflEpcParams);
    return config;
}

OptixPipelineConfig reflection_epc_direct_primary_pipeline_config() {
    OptixPipelineConfig config;
    config.ptx = reflection_epc_ptx;
    config.ptx_size = reflection_epc_ptx_size;
    config.raygen_entries = {"__raygen__reflection_epc_direct_primary"};
    config.miss_entry = "__miss__reflection_epc";
    config.closesthit_entry = "__closesthit__reflection_epc";
    config.anyhit_entry = "__anyhit__reflection_epc";
    config.num_payload_values = shared::optix::TriangleHitPayloadCount;
    config.params_size = sizeof(ReflEpcParams);
    return config;
}

OptixPipelineConfig reflection_accumulation_pipeline_config() {
    OptixPipelineConfig config;
    config.ptx = reflection_accumulation_ptx;
    config.ptx_size = reflection_accumulation_ptx_size;
    config.raygen_entries = {"__raygen__reflection_accumulation"};
    config.miss_entry = "__miss__reflection_accumulation";
    config.closesthit_entry = "__closesthit__reflection_accumulation";
    config.num_payload_values = shared::optix::TriangleHitPayloadCount;
    config.params_size = sizeof(AccumParams);
    return config;
}

OptixPipelineConfig diffraction_accumulation_pipeline_config() {
    OptixPipelineConfig config;
    config.ptx = diffraction_accumulation_ptx;
    config.ptx_size = diffraction_accumulation_ptx_size;
    config.raygen_entries = {
        "__raygen__diffraction_order1_accumulation",
        "__raygen__diffraction_chain_accumulation",
        "__raygen__diffraction_order1_coherent_accumulation",
    };
    config.miss_entry = "__miss__diffraction_accumulation";
    config.closesthit_entry = "__closesthit__diffraction_accumulation";
    config.num_payload_values = shared::optix::DiffractionPayloadCount;
    config.params_size = sizeof(DfrAccumParams);
    return config;
}

OptixPipelineConfig diffraction_order1_accumulation_pipeline_config() {
    OptixPipelineConfig config = diffraction_accumulation_pipeline_config();
    config.raygen_entries = {"__raygen__diffraction_order1_accumulation"};
    return config;
}

OptixPipelineConfig diffraction_order1_accumulation_primary_pipeline_config() {
    OptixPipelineConfig config = diffraction_accumulation_pipeline_config();
    config.raygen_entries = {"__raygen__diffraction_order1_accumulation_primary"};
    return config;
}

OptixPipelineConfig diffraction_order1_accumulation_no_suffix_pipeline_config() {
    OptixPipelineConfig config = diffraction_accumulation_pipeline_config();
    config.raygen_entries = {"__raygen__diffraction_order1_accumulation_no_suffix"};
    return config;
}

OptixPipelineConfig diffraction_order1_accumulation_no_suffix_primary_pipeline_config() {
    OptixPipelineConfig config = diffraction_accumulation_pipeline_config();
    config.raygen_entries = {"__raygen__diffraction_order1_accumulation_no_suffix_primary"};
    return config;
}

OptixPipelineConfig diffraction_order1_accumulation_suffix_pipeline_config() {
    OptixPipelineConfig config = diffraction_accumulation_pipeline_config();
    config.raygen_entries = {"__raygen__diffraction_order1_accumulation_suffix"};
    return config;
}

OptixPipelineConfig diffraction_order1_accumulation_suffix_primary_pipeline_config() {
    OptixPipelineConfig config = diffraction_accumulation_pipeline_config();
    config.raygen_entries = {"__raygen__diffraction_order1_accumulation_suffix_primary"};
    return config;
}

OptixPipelineConfig diffraction_order1_source_visibility_primary_pipeline_config() {
    OptixPipelineConfig config = diffraction_accumulation_pipeline_config();
    config.raygen_entries = {"__raygen__diffraction_order1_source_visibility_primary"};
    return config;
}

OptixPipelineConfig diffraction_order1_no_suffix_target_primary_pipeline_config() {
    OptixPipelineConfig config = diffraction_accumulation_pipeline_config();
    config.raygen_entries = {"__raygen__diffraction_order1_no_suffix_target_accumulation_primary"};
    return config;
}

OptixPipelineConfig diffraction_order1_suffix_first_visibility_primary_pipeline_config() {
    OptixPipelineConfig config = diffraction_accumulation_pipeline_config();
    config.raygen_entries = {"__raygen__diffraction_order1_suffix_first_visibility_primary"};
    return config;
}

OptixPipelineConfig diffraction_order1_suffix_target_primary_pipeline_config() {
    OptixPipelineConfig config = diffraction_accumulation_pipeline_config();
    config.raygen_entries = {"__raygen__diffraction_order1_suffix_target_accumulation_primary"};
    return config;
}

OptixPipelineConfig diffraction_chain_accumulation_pipeline_config() {
    OptixPipelineConfig config = diffraction_accumulation_pipeline_config();
    config.raygen_entries = {"__raygen__diffraction_chain_accumulation"};
    return config;
}

OptixPipelineConfig diffraction_chain_accumulation_primary_pipeline_config() {
    OptixPipelineConfig config = diffraction_accumulation_pipeline_config();
    config.raygen_entries = {"__raygen__diffraction_chain_accumulation_primary"};
    return config;
}

OptixPipelineConfig diffraction_coherent_accumulation_pipeline_config() {
    OptixPipelineConfig config = diffraction_accumulation_pipeline_config();
    config.raygen_entries = {"__raygen__diffraction_order1_coherent_accumulation"};
    return config;
}

OptixPipelineConfig diffraction_coherent_accumulation_primary_pipeline_config() {
    OptixPipelineConfig config = diffraction_accumulation_pipeline_config();
    config.raygen_entries = {"__raygen__diffraction_order1_coherent_accumulation_primary"};
    return config;
}

OptixPipelineConfig diffraction_paths_pipeline_config() {
    OptixPipelineConfig config;
    config.ptx = diffraction_paths_ptx;
    config.ptx_size = diffraction_paths_ptx_size;
    config.raygen_entries = {"__raygen__diffraction_paths_order1"};
    config.miss_entry = "__miss__diffraction_paths";
    config.closesthit_entry = "__closesthit__diffraction_paths";
    config.num_payload_values = shared::optix::DiffractionPayloadCount;
    config.params_size = sizeof(DfrPathParams);
    return config;
}

OptixPipelineConfig diffraction_paths_primary_pipeline_config() {
    OptixPipelineConfig config = diffraction_paths_pipeline_config();
    config.raygen_entries = {"__raygen__diffraction_paths_order1_primary"};
    return config;
}

OptixPipelineConfig diffraction_paths_source_visibility_primary_pipeline_config() {
    OptixPipelineConfig config = diffraction_paths_pipeline_config();
    config.raygen_entries = {"__raygen__diffraction_paths_order1_source_visibility_primary"};
    return config;
}

OptixPipelineConfig diffraction_paths_target_export_primary_pipeline_config() {
    OptixPipelineConfig config = diffraction_paths_pipeline_config();
    config.raygen_entries = {"__raygen__diffraction_paths_order1_target_export_primary"};
    return config;
}

OptixPipelineConfig segment_visibility_pipeline_config() {
    OptixPipelineConfig config;
    config.ptx = segment_visibility_ptx;
    config.ptx_size = segment_visibility_ptx_size;
    config.raygen_entries = {"__raygen__segment_visibility"};
    config.miss_entry = "__miss__segment_visibility";
    config.closesthit_entry = "__closesthit__segment_visibility";
    config.anyhit_entry = "__anyhit__segment_visibility";
    config.num_payload_values = shared::optix::VisibilityPayloadCount;
    config.params_size = sizeof(SegmentVisibilityParams);
    return config;
}

OptixPipelineConfig segment_pair_visibility_pipeline_config() {
    OptixPipelineConfig config = segment_visibility_pipeline_config();
    config.raygen_entries = {"__raygen__segment_pair_visibility"};
    return config;
}

OptixPipelineConfig axial_edge_visibility_pipeline_config() {
    OptixPipelineConfig config = segment_visibility_pipeline_config();
    config.raygen_entries = {"__raygen__axial_edge_visibility"};
    return config;
}

OptixPipelineConfig segment_chain_visibility_pipeline_config() {
    OptixPipelineConfig config = segment_visibility_pipeline_config();
    config.raygen_entries = {"__raygen__segment_chain_visibility"};
    return config;
}

} // namespace rayd

// Consolidated multipath configuration host facade.
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <exception>
#include <limits>
#include <string>
#include <vector>

#include <rayd/jit/ray.h>
#include "scene_internal_jit.h"
#include <src/diffraction/accumulation_ad_jit.h>
#include <src/reflection/dedup_jit.h>
#include <src/reflection/epc_field_jit.h>
#include <src/runtime/optix_pipelines_jit.h>
#include <rayd/jit/native_launch_audit.h>
#include <src/scene/cuda_multipath_gpu_jit.h>

#include "multipath_internal_jit.h"

namespace rayd {

namespace {

std::string normalize_optix_split_mode_value(const char *value) {
    std::string normalized = value != nullptr ? std::string(value) : std::string();
    std::transform(normalized.begin(),
                   normalized.end(),
                   normalized.begin(),
                   [](unsigned char ch) -> char {
                       return static_cast<char>(std::tolower(ch));
                   });
    return normalized;
}

} // namespace

namespace multipath_detail {

TraceVisibilityBackend active_trace_visibility_backend() {
    static const TraceVisibilityBackend value = []() {
        const char *raw = std::getenv("RAYD_TRACE_VISIBILITY_BACKEND");
        const std::string normalized = normalize_optix_split_mode_value(raw);
        if (normalized.empty() || normalized == "auto") {
            return TraceVisibilityBackend::Auto;
        }
        if (normalized == "jit" || normalized == "drjit" ||
            normalized == "hitobject" || normalized == "hit_object") {
            return TraceVisibilityBackend::Jit;
        }
        if (normalized == "native" || normalized == "optixlaunch" ||
            normalized == "optix_launch") {
            return TraceVisibilityBackend::Native;
        }
        throw std::runtime_error(
            "Invalid RAYD_TRACE_VISIBILITY_BACKEND. Expected one of: auto, jit, native.");
    }();
    return value;
}

ReflEpcVisibilityIgnoreMode parse_refl_epc_vis_ignore(
    const std::string &value) {
    const std::string normalized = normalize_optix_split_mode_value(value.c_str());
    if (normalized.empty() || normalized == "primitive" ||
        normalized == "prim" || normalized == "exact") {
        return ReflEpcVisibilityIgnoreMode::Primitive;
    }
    if (normalized == "surface_group" || normalized == "surface-group" ||
        normalized == "group") {
        return ReflEpcVisibilityIgnoreMode::SurfaceGroup;
    }
    throw std::runtime_error(
        "Invalid ReflEpcOptions.visibility_ignore_mode. "
        "Expected one of: 'primitive', 'surface_group'.");
}

bool use_jit_trace_visibility_path(int ignore_k) {
    const TraceVisibilityBackend backend = active_trace_visibility_backend();
    if (backend == TraceVisibilityBackend::Native) {
        return false;
    }
    if (backend == TraceVisibilityBackend::Jit) {
        require(ignore_k == 0,
                "RAYD_TRACE_VISIBILITY_BACKEND=jit does not support ignore lists yet.");
        return true;
    }
    return ignore_k == 0;
}

bool recording_reflections() {
    return jit_flag(JitFlag::Recording);
}

bool uses_symbolic_optix_query_path() {
    // Dr.Jit symbolic recording cannot mix multiple OptiX pipelines/SBTs
    // within a single captured kernel. Fall back to the unified scene path.
    return jit_flag(JitFlag::Recording);
}

void ensure_pipeline(std::shared_ptr<OptixLaunchPipeline> &pipeline,
                     OptixDeviceContext context,
                     int hitgroup_record_count,
                     const OptixPipelineConfig &config) {
    if (!pipeline) {
        pipeline = shared_optix_launch_pipeline(context, hitgroup_record_count, config);
    }
}

} // namespace multipath_detail

} // namespace rayd
