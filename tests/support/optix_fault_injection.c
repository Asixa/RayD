// Copyright Xingyu Chen.
// Wraps the OptiX driver library so tests can make chosen OptiX pipeline-building calls fail.

// Build as a shared object named libnvoptix.so.1 and put its directory first on LD_LIBRARY_PATH: both backends
// then load this file instead of the driver library, and every function table they query comes from the real
// driver (named by RAYD_FAULT_REAL_OPTIX) with three slots replaced: optixPipelineCreate (fails with a link error,
// the compile-class failure RayD retries), optixPipelineSetStackSize (fails with an invalid-value error, a failure
// RayD must not retry, raised after the pipeline object exists), and optixDeviceContextSetCacheEnabled (counted).
// Slots are found by value, by comparing against a reference table of a known ABI, so the wrapper works for
// whichever ABI version each backend requests. Nothing here changes RayD itself.

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int OptixResult;
typedef OptixResult (*QueryFunctionTableFn)(int, unsigned int, void*, const void**, void*, size_t);
typedef OptixResult (*PipelineCreateFn)(void*, const void*, const void*, const void*, unsigned int, char*, size_t*,
                                        void**);
typedef OptixResult (*SetStackSizeFn)(void*, unsigned int, unsigned int, unsigned int, unsigned int);
typedef OptixResult (*SetCacheEnabledFn)(void*, int);
typedef void* (*SymbolTableFn)(void);

enum {
    kReferenceAbi = 118,
    kReferenceSlotCount = 60,
    kSetCacheEnabledSlot = 6,
    kPipelineCreateSlot = 27,
    kPipelineSetStackSizeSlot = 30,
    kInvalidValue = 7001,
    kPipelineLinkError = 7251,
    kLibraryNotFound = 7804,
};

typedef struct {
    int skip;
    int fail;
    int calls;
    int injected;
} FaultCounter;

static void* real_library = NULL;
static QueryFunctionTableFn real_query = NULL;
static PipelineCreateFn real_pipeline_create = NULL;
static SetStackSizeFn real_set_stack_size = NULL;
static SetCacheEnabledFn real_set_cache_enabled = NULL;
static FaultCounter pipeline_create = {0, 0, 0, 0};
static FaultCounter set_stack_size = {0, 0, 0, 0};
static int cache_disable_calls = 0;
static int cache_enable_calls = 0;
static int patched_tables = 0;

static int load_real(void) {
    if (real_query != NULL)
        return 1;
    const char* path = getenv("RAYD_FAULT_REAL_OPTIX");
    if (path == NULL || path[0] == '\0') {
        fprintf(stderr, "optix_fault_injection: RAYD_FAULT_REAL_OPTIX is not set\n");
        return 0;
    }
    real_library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (real_library == NULL) {
        fprintf(stderr, "optix_fault_injection: cannot load %s: %s\n", path, dlerror());
        return 0;
    }
    QueryFunctionTableFn query = (QueryFunctionTableFn)dlsym(real_library, "optixQueryFunctionTable");
    if (query == NULL)
        return 0;
    void* reference[kReferenceSlotCount];
    memset(reference, 0, sizeof(reference));
    if (query(kReferenceAbi, 0, NULL, NULL, reference, sizeof(reference)) != 0) {
        fprintf(stderr, "optix_fault_injection: the driver does not provide reference ABI %d\n", kReferenceAbi);
        return 0;
    }
    real_pipeline_create = (PipelineCreateFn)reference[kPipelineCreateSlot];
    real_set_stack_size = (SetStackSizeFn)reference[kPipelineSetStackSizeSlot];
    real_set_cache_enabled = (SetCacheEnabledFn)reference[kSetCacheEnabledSlot];
    real_query = query;
    return 1;
}

// Returns 1 when this call must fail, after counting it.
static int take_fault(FaultCounter* counter) {
    ++counter->calls;
    if (counter->skip > 0) {
        --counter->skip;
        return 0;
    }
    if (counter->fail > 0) {
        --counter->fail;
        ++counter->injected;
        return 1;
    }
    return 0;
}

static OptixResult fault_pipeline_create(void* context, const void* compile_options, const void* link_options,
                                         const void* program_groups, unsigned int group_count, char* log,
                                         size_t* log_size, void** pipeline) {
    if (take_fault(&pipeline_create)) {
        if (pipeline != NULL)
            *pipeline = NULL;
        if (log != NULL && log_size != NULL && *log_size > 0) {
            const int written = snprintf(log, *log_size, "injected optixPipelineCreate link failure");
            *log_size = written > 0 ? (size_t)written + 1 : 0;
        }
        return kPipelineLinkError;
    }
    return real_pipeline_create(context, compile_options, link_options, program_groups, group_count, log, log_size,
                                pipeline);
}

static OptixResult fault_set_stack_size(void* pipeline, unsigned int from_traversal, unsigned int from_state,
                                        unsigned int continuation, unsigned int max_traversable_depth) {
    if (take_fault(&set_stack_size))
        return kInvalidValue;
    return real_set_stack_size(pipeline, from_traversal, from_state, continuation, max_traversable_depth);
}

static OptixResult fault_set_cache_enabled(void* context, int enabled) {
    if (enabled)
        ++cache_enable_calls;
    else
        ++cache_disable_calls;
    return real_set_cache_enabled(context, enabled);
}

OptixResult optixQueryFunctionTable(int abi, unsigned int option_count, void* option_keys, const void** option_values,
                                    void* table, size_t table_size) {
    if (!load_real())
        return kLibraryNotFound;
    const OptixResult result = real_query(abi, option_count, option_keys, option_values, table, table_size);
    if (result != 0 || table == NULL)
        return result;
    void** slots = (void**)table;
    const size_t slot_count = table_size / sizeof(void*);
    int patched = 0;
    for (size_t i = 0; i < slot_count; ++i) {
        if (slots[i] == (void*)real_pipeline_create) {
            slots[i] = (void*)&fault_pipeline_create;
            patched |= 1;
        } else if (slots[i] == (void*)real_set_stack_size) {
            slots[i] = (void*)&fault_set_stack_size;
            patched |= 2;
        } else if (slots[i] == (void*)real_set_cache_enabled) {
            slots[i] = (void*)&fault_set_cache_enabled;
            patched |= 4;
        }
    }
    if (patched == 7)
        ++patched_tables;
    return result;
}

void* rtGetSymbolTable(void) {
    if (!load_real())
        return NULL;
    SymbolTableFn real = (SymbolTableFn)dlsym(real_library, "rtGetSymbolTable");
    return real != NULL ? real() : NULL;
}

// Controls the test reads and writes through ctypes. Arming replaces any earlier arming of the same call.

void rayd_fault_arm_pipeline_create(int skip, int fail) {
    pipeline_create.skip = skip;
    pipeline_create.fail = fail;
}

void rayd_fault_arm_set_stack_size(int skip, int fail) {
    set_stack_size.skip = skip;
    set_stack_size.fail = fail;
}

int rayd_fault_pipeline_create_calls(void) {
    return pipeline_create.calls;
}

int rayd_fault_pipeline_create_injected(void) {
    return pipeline_create.injected;
}

int rayd_fault_set_stack_size_calls(void) {
    return set_stack_size.calls;
}

int rayd_fault_set_stack_size_injected(void) {
    return set_stack_size.injected;
}

int rayd_fault_cache_disable_calls(void) {
    return cache_disable_calls;
}

int rayd_fault_cache_enable_calls(void) {
    return cache_enable_calls;
}

int rayd_fault_patched_tables(void) {
    return patched_tables;
}
