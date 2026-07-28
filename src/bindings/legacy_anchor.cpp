#if defined(_WIN32)
#  define RAYD_TORCH_LEGACY_EXPORT __declspec(dllexport)
#else
#  define RAYD_TORCH_LEGACY_EXPORT __attribute__((visibility("default")))
#endif

extern "C" RAYD_TORCH_LEGACY_EXPORT int rayd_torch_legacy_ops_anchor() {
    return 1;
}
