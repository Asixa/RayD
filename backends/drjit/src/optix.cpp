#include <drjit-core/optix.h>
#define OPTIX_STUBS_IMPL
#include <rayd/optix.h>

#include <rayd/native_launch_audit.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
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

bool optix_available() noexcept {
    static const bool available = []() noexcept -> bool {
        try {
            if (env_disables_optix())
                return false;

            auto module = probe_load_optix_module();
            if (!module)
                return false;

            OptixQueryFunctionTableFn query_fn = optix_query_function_table(module);
            if (query_fn == nullptr)
                return false;

            // ABI-93 probe (mirrors query_optix_runtime_info): 7801 signals the
            // driver rejects the target ABI, anything else means it is supported.
            const OptixResult abi_probe =
                query_fn(RAYD_OPTIX_TARGET_ABI, 0, nullptr, nullptr, nullptr, 0);
            return abi_probe != 7801;
        } catch (...) {
            return false;
        }
    }();
    return available;
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
