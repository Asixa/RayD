#include <drjit-core/optix.h>
#define OPTIX_STUBS_IMPL
#include <rayd/optix.h>

#include <sstream>

#if defined(_WIN32)
#  define NOMINMAX
#  include <windows.h>
#  include <winver.h>
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
    L(optixSbtRecordPackHeader);

    #undef L
}
