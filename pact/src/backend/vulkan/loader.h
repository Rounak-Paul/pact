#ifndef PACT_VULKAN_LOADER_H
#define PACT_VULKAN_LOADER_H

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>

#include <pact/pact.h>

typedef struct pact_vulkan_state {
    void *library;
    VkInstance instance;
    PFN_vkGetInstanceProcAddr get_instance_proc_addr;
    PFN_vkEnumerateInstanceExtensionProperties enumerate_instance_extensions;
    PFN_vkCreateInstance create_instance;
    PFN_vkDestroyInstance destroy_instance;
    PFN_vkEnumeratePhysicalDevices enumerate_physical_devices;
    PFN_vkGetPhysicalDeviceProperties get_physical_device_properties;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties get_queue_family_properties;
} pact_vulkan_state;

pact_result pact_vulkan_loader_open(pact_vulkan_state **out_state);
pact_result pact_vulkan_loader_load_instance(pact_vulkan_state *state);
void pact_vulkan_loader_close(pact_vulkan_state *state);

#endif
