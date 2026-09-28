#include "runtime_internal.h"
#include "loader.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int pact_vk_allocation_size(uint32_t count, size_t item_size, size_t *out_size)
{
    size_t item_count = (size_t)count;
    if (item_size != 0 && item_count > SIZE_MAX / item_size) {
        return 0;
    }
    *out_size = item_count * item_size;
    return 1;
}

static pact_result pact_vk_result(VkResult result)
{
    if (result == VK_ERROR_OUT_OF_HOST_MEMORY || result == VK_ERROR_OUT_OF_DEVICE_MEMORY) {
        return PACT_ERROR_OUT_OF_MEMORY;
    }
    return PACT_ERROR_BACKEND_UNAVAILABLE;
}

#if defined(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)
static pact_result pact_vk_has_portability_enumeration(
    pact_vulkan_state *state, int *out_found)
{
    *out_found = 0;
    for (unsigned int attempt = 0; attempt < 4; ++attempt) {
        uint32_t count = 0;
        VkResult result = state->enumerate_instance_extensions(NULL, &count, NULL);
        if (result != VK_SUCCESS) {
            return pact_vk_result(result);
        }
        if (count == 0) {
            return PACT_SUCCESS;
        }
        size_t allocation_size = 0;
        if (!pact_vk_allocation_size(count, sizeof(VkExtensionProperties), &allocation_size)) {
            return PACT_ERROR_OUT_OF_MEMORY;
        }
        VkExtensionProperties *extensions = (VkExtensionProperties *)malloc(allocation_size);
        if (extensions == NULL) {
            return PACT_ERROR_OUT_OF_MEMORY;
        }
        result = state->enumerate_instance_extensions(NULL, &count, extensions);
        if (result == VK_INCOMPLETE) {
            free(extensions);
            continue;
        }
        if (result != VK_SUCCESS) {
            free(extensions);
            return pact_vk_result(result);
        }
        for (uint32_t i = 0; i < count; ++i) {
            if (strcmp(extensions[i].extensionName,
                       VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) == 0) {
                *out_found = 1;
                break;
            }
        }
        free(extensions);
        return PACT_SUCCESS;
    }
    return PACT_ERROR_INTERNAL;
}
#endif

static pact_result pact_vk_supports_compute(
    pact_vulkan_state *state, VkPhysicalDevice device, int *out_supported)
{
    *out_supported = 0;
    uint32_t count = 0;
    state->get_queue_family_properties(device, &count, NULL);
    size_t allocation_size = 0;
    if (count == 0) {
        return PACT_SUCCESS;
    }
    if (!pact_vk_allocation_size(count, sizeof(VkQueueFamilyProperties), &allocation_size)) {
        return PACT_ERROR_OUT_OF_MEMORY;
    }
    VkQueueFamilyProperties *families = (VkQueueFamilyProperties *)malloc(allocation_size);
    if (families == NULL) {
        return PACT_ERROR_OUT_OF_MEMORY;
    }
    state->get_queue_family_properties(device, &count, families);
    for (uint32_t i = 0; i < count; ++i) {
        if (families[i].queueCount > 0 && (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
            *out_supported = 1;
            break;
        }
    }
    free(families);
    return PACT_SUCCESS;
}

static pact_result pact_vk_add_device(
    pact_runtime *runtime, pact_vulkan_state *state, VkPhysicalDevice device)
{
    VkPhysicalDeviceProperties properties;
    state->get_physical_device_properties(device, &properties);
    /* A software Vulkan CPU would duplicate the native CPU execution device. */
    if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) {
        return PACT_SUCCESS;
    }
    int supports_compute = 0;
    pact_result status = pact_vk_supports_compute(state, device, &supports_compute);
    if (status != PACT_SUCCESS || !supports_compute) {
        return status;
    }

    pact_device_info info;
    memset(&info, 0, sizeof(info));
    info.struct_size = sizeof(info);
    info.backend = PACT_BACKEND_VULKAN;
    info.type = properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_OTHER
                    ? PACT_DEVICE_TYPE_OTHER
                    : PACT_DEVICE_TYPE_GPU;
    snprintf(info.name, sizeof(info.name), "%s", properties.deviceName);
    info.vendor_id = properties.vendorID;
    info.device_id = properties.deviceID;
    info.backend_api_version = properties.apiVersion;
    return pact_runtime_add_device(runtime, &info, (void *)device);
}

pact_result pact_vulkan_discover(pact_runtime *runtime)
{
    pact_vulkan_state *state = NULL;
    pact_result loader_status = pact_vulkan_loader_open(&state);
    if (loader_status != PACT_SUCCESS) {
        return loader_status;
    }
    runtime->vulkan_state = state;

    VkApplicationInfo application = {0};
    application.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    application.pApplicationName = "PACT";
    application.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    application.pEngineName = "PACT";
    application.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    application.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo create_info = {0};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &application;

#if defined(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)
    const char *portability_extension = VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME;
    int has_portability = 0;
    pact_result portability_status = pact_vk_has_portability_enumeration(state, &has_portability);
    if (portability_status != PACT_SUCCESS) {
        return portability_status;
    }
    if (has_portability) {
        create_info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
        create_info.enabledExtensionCount = 1;
        create_info.ppEnabledExtensionNames = &portability_extension;
    }
#endif

    VkInstance instance = VK_NULL_HANDLE;
    VkResult result = state->create_instance(&create_info, NULL, &instance);
    if (result != VK_SUCCESS) {
        return pact_vk_result(result);
    }
    state->instance = instance;
    pact_result instance_status = pact_vulkan_loader_load_instance(state);
    if (instance_status != PACT_SUCCESS) {
        return instance_status;
    }

    VkPhysicalDevice *devices = NULL;
    uint32_t count = 0;
    for (unsigned int attempt = 0; attempt < 4; ++attempt) {
        result = state->enumerate_physical_devices(instance, &count, NULL);
        if (result != VK_SUCCESS) {
            return pact_vk_result(result);
        }
        if (count == 0) {
            return PACT_ERROR_BACKEND_UNAVAILABLE;
        }
        size_t allocation_size = 0;
        if (!pact_vk_allocation_size(count, sizeof(VkPhysicalDevice), &allocation_size)) {
            return PACT_ERROR_OUT_OF_MEMORY;
        }
        devices = (VkPhysicalDevice *)malloc(allocation_size);
        if (devices == NULL) {
            return PACT_ERROR_OUT_OF_MEMORY;
        }
        result = state->enumerate_physical_devices(instance, &count, devices);
        if (result == VK_INCOMPLETE) {
            free(devices);
            devices = NULL;
            continue;
        }
        if (result != VK_SUCCESS) {
            free(devices);
            return pact_vk_result(result);
        }
        break;
    }
    if (devices == NULL) {
        return PACT_ERROR_INTERNAL;
    }

    uint32_t before = runtime->device_count;
    pact_result status = PACT_SUCCESS;
    for (uint32_t i = 0; i < count; ++i) {
        status = pact_vk_add_device(runtime, state, devices[i]);
        if (status != PACT_SUCCESS) {
            break;
        }
    }
    free(devices);
    if (status != PACT_SUCCESS) {
        return status;
    }
    return runtime->device_count > before ? PACT_SUCCESS : PACT_ERROR_BACKEND_UNAVAILABLE;
}

void pact_vulkan_release(pact_runtime *runtime)
{
    pact_vulkan_loader_close((pact_vulkan_state *)runtime->vulkan_state);
    runtime->vulkan_state = NULL;
}
