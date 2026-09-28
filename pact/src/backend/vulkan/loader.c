#include "loader.h"

#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

static void *pact_vulkan_open_library(void)
{
#if defined(_WIN32)
    return (void *)LoadLibraryA("vulkan-1.dll");
#elif defined(__APPLE__)
    const char *names[] = {
        "libvulkan.1.dylib",
        "libvulkan.dylib",
        "libMoltenVK.dylib"
    };
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
        void *library = dlopen(names[i], RTLD_NOW | RTLD_LOCAL);
        if (library != NULL) {
            return library;
        }
    }
    return NULL;
#else
    void *library = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    if (library == NULL) {
        library = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
    }
    return library;
#endif
}

static void pact_vulkan_close_library(void *library)
{
    if (library == NULL) {
        return;
    }
#if defined(_WIN32)
    FreeLibrary((HMODULE)library);
#else
    dlclose(library);
#endif
}

static PFN_vkGetInstanceProcAddr pact_vulkan_find_get_proc_addr(void *library)
{
    PFN_vkGetInstanceProcAddr proc = NULL;
#if defined(_WIN32)
    FARPROC symbol = GetProcAddress((HMODULE)library, "vkGetInstanceProcAddr");
#else
    void *symbol = dlsym(library, "vkGetInstanceProcAddr");
#endif
    if (symbol != NULL && sizeof(proc) == sizeof(symbol)) {
        memcpy(&proc, &symbol, sizeof(proc));
    }
    return proc;
}

static int pact_vulkan_load_proc(
    pact_vulkan_state *state, VkInstance instance, const char *name, void *out_proc,
    size_t out_size)
{
    PFN_vkVoidFunction proc = state->get_instance_proc_addr(instance, name);
    if (proc == NULL || out_size != sizeof(proc)) {
        return 0;
    }
    memcpy(out_proc, &proc, out_size);
    return 1;
}

pact_result pact_vulkan_loader_open(pact_vulkan_state **out_state)
{
    *out_state = NULL;
    pact_vulkan_state *state = (pact_vulkan_state *)calloc(1, sizeof(*state));
    if (state == NULL) {
        return PACT_ERROR_OUT_OF_MEMORY;
    }
    state->library = pact_vulkan_open_library();
    if (state->library == NULL) {
        free(state);
        return PACT_ERROR_BACKEND_UNAVAILABLE;
    }
    state->get_instance_proc_addr = pact_vulkan_find_get_proc_addr(state->library);
    if (state->get_instance_proc_addr == NULL) {
        pact_vulkan_loader_close(state);
        return PACT_ERROR_BACKEND_UNAVAILABLE;
    }
    if (!pact_vulkan_load_proc(
            state, VK_NULL_HANDLE, "vkEnumerateInstanceExtensionProperties",
            &state->enumerate_instance_extensions, sizeof(state->enumerate_instance_extensions)) ||
        !pact_vulkan_load_proc(
            state, VK_NULL_HANDLE, "vkCreateInstance",
            &state->create_instance, sizeof(state->create_instance))) {
        pact_vulkan_loader_close(state);
        return PACT_ERROR_INTERNAL;
    }
    *out_state = state;
    return PACT_SUCCESS;
}

pact_result pact_vulkan_loader_load_instance(pact_vulkan_state *state)
{
    if (!pact_vulkan_load_proc(
            state, state->instance, "vkDestroyInstance",
            &state->destroy_instance, sizeof(state->destroy_instance)) ||
        !pact_vulkan_load_proc(
            state, state->instance, "vkEnumeratePhysicalDevices",
            &state->enumerate_physical_devices, sizeof(state->enumerate_physical_devices)) ||
        !pact_vulkan_load_proc(
            state, state->instance, "vkGetPhysicalDeviceProperties",
            &state->get_physical_device_properties,
            sizeof(state->get_physical_device_properties)) ||
        !pact_vulkan_load_proc(
            state, state->instance, "vkGetPhysicalDeviceQueueFamilyProperties",
            &state->get_queue_family_properties, sizeof(state->get_queue_family_properties))) {
        return PACT_ERROR_INTERNAL;
    }
    return PACT_SUCCESS;
}

void pact_vulkan_loader_close(pact_vulkan_state *state)
{
    if (state == NULL) {
        return;
    }
    if (state->instance != VK_NULL_HANDLE && state->destroy_instance != NULL) {
        state->destroy_instance(state->instance, NULL);
    }
    pact_vulkan_close_library(state->library);
    free(state);
}
