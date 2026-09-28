#include "tensor_internal.h"
#include "loader.h"
#include "shaders/graph_f32_spv.h"
#include "shaders/linear_f32_spv.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

#define VK_FUNCTIONS(X) \
    X(vkCreateDevice) X(vkGetDeviceProcAddr) X(vkGetPhysicalDeviceMemoryProperties) \
    X(vkEnumerateDeviceExtensionProperties) X(vkGetDeviceQueue) X(vkDestroyDevice) \
    X(vkCreateBuffer) X(vkDestroyBuffer) X(vkGetBufferMemoryRequirements) \
    X(vkAllocateMemory) X(vkFreeMemory) X(vkBindBufferMemory) \
    X(vkMapMemory) X(vkUnmapMemory) X(vkCreateShaderModule) X(vkDestroyShaderModule) \
    X(vkCreateDescriptorSetLayout) X(vkDestroyDescriptorSetLayout) \
    X(vkCreatePipelineLayout) X(vkDestroyPipelineLayout) \
    X(vkCreateComputePipelines) X(vkDestroyPipeline) \
    X(vkCreateDescriptorPool) X(vkDestroyDescriptorPool) \
    X(vkAllocateDescriptorSets) X(vkUpdateDescriptorSets) \
    X(vkCreateCommandPool) X(vkDestroyCommandPool) X(vkResetCommandPool) \
    X(vkAllocateCommandBuffers) X(vkBeginCommandBuffer) X(vkEndCommandBuffer) \
    X(vkCmdCopyBuffer) X(vkCmdPipelineBarrier) X(vkCmdBindPipeline) \
    X(vkCmdBindDescriptorSets) X(vkCmdPushConstants) X(vkCmdDispatch) \
    X(vkCreateFence) X(vkDestroyFence) X(vkResetFences) X(vkQueueSubmit) \
    X(vkWaitForFences) X(vkDeviceWaitIdle)

#define DECLARE(name) PFN_##name name;
typedef struct pact_vk_functions { VK_FUNCTIONS(DECLARE) } pact_vk_functions;
#undef DECLARE

typedef struct pact_vk_pair {
    VkBuffer first;
    VkBuffer second;
    VkDeviceMemory memory;
    void *mapped;
    VkDeviceSize second_offset;
} pact_vk_pair;

typedef struct pact_vulkan_context {
    pact_vk_functions fn;
    VkDevice device;
    VkPhysicalDevice physical;
    VkQueue queue;
    VkPhysicalDeviceProperties properties;
    VkPhysicalDeviceMemoryProperties memory_properties;
    VkDescriptorSetLayout descriptor_layout;
    VkPipelineLayout pipeline_layout;
    VkPipeline pipeline;
    VkPipeline linear_pipeline;
    VkDescriptorPool descriptor_pool;
    VkDescriptorSet descriptor_set;
    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;
    VkFence fence;
    pact_vk_pair device_data;
    pact_vk_pair staging;
    VkBuffer program_buffer;
    VkDeviceMemory program_memory;
    VkBuffer program_staging_buffer;
    VkDeviceMemory program_staging_memory;
    void *program_mapped;
    int program_direct;
    size_t capacity;
    size_t source_capacity;
    int direct_access;
    int failed;
} pact_vulkan_context;

static double pact_now_us(void)
{
    struct timespec time;
    timespec_get(&time, TIME_UTC);
    return (double)time.tv_sec * 1000000.0 + (double)time.tv_nsec / 1000.0;
}

static pact_result pact_vk_error(VkResult result)
{
    if (result == VK_ERROR_OUT_OF_HOST_MEMORY || result == VK_ERROR_OUT_OF_DEVICE_MEMORY)
        return PACT_ERROR_OUT_OF_MEMORY;
    return PACT_ERROR_BACKEND_UNAVAILABLE;
}

static int pact_vk_memory_type(
    const VkPhysicalDeviceMemoryProperties *memory, uint32_t bits,
    VkMemoryPropertyFlags required, uint32_t *out_type)
{
    for (uint32_t i = 0; i < memory->memoryTypeCount; ++i) {
        if ((bits & (1u << i)) &&
            (memory->memoryTypes[i].propertyFlags & required) == required) {
            *out_type = i;
            return 1;
        }
    }
    return 0;
}

static int pact_vk_align(VkDeviceSize value, VkDeviceSize alignment, VkDeviceSize *out)
{
    if (alignment == 0) return 0;
    VkDeviceSize remainder = value % alignment;
    VkDeviceSize padding = remainder == 0 ? 0 : alignment - remainder;
    if (value > UINT64_MAX - padding) return 0;
    *out = value + padding;
    return 1;
}

static void pact_vk_pair_destroy(pact_vulkan_context *context, pact_vk_pair *pair)
{
    if (pair->mapped != NULL) context->fn.vkUnmapMemory(context->device, pair->memory);
    if (pair->first != VK_NULL_HANDLE)
        context->fn.vkDestroyBuffer(context->device, pair->first, NULL);
    if (pair->second != VK_NULL_HANDLE)
        context->fn.vkDestroyBuffer(context->device, pair->second, NULL);
    if (pair->memory != VK_NULL_HANDLE)
        context->fn.vkFreeMemory(context->device, pair->memory, NULL);
    *pair = (pact_vk_pair){0};
}

void pact_vulkan_context_destroy(void *opaque)
{
    pact_vulkan_context *context = opaque;
    if (context == NULL) return;
    if (context->device != VK_NULL_HANDLE) {
        if (context->fn.vkDeviceWaitIdle != NULL)
            context->fn.vkDeviceWaitIdle(context->device);
        if (context->fn.vkDestroyBuffer != NULL) {
            pact_vk_pair_destroy(context, &context->device_data);
            pact_vk_pair_destroy(context, &context->staging);
            if (context->program_mapped != NULL)
                context->fn.vkUnmapMemory(context->device,
                    context->program_direct ? context->program_memory
                                            : context->program_staging_memory);
            if (context->program_staging_buffer != VK_NULL_HANDLE)
                context->fn.vkDestroyBuffer(
                    context->device, context->program_staging_buffer, NULL);
            if (context->program_staging_memory != VK_NULL_HANDLE)
                context->fn.vkFreeMemory(
                    context->device, context->program_staging_memory, NULL);
            if (context->program_buffer != VK_NULL_HANDLE)
                context->fn.vkDestroyBuffer(context->device, context->program_buffer, NULL);
            if (context->program_memory != VK_NULL_HANDLE)
                context->fn.vkFreeMemory(context->device, context->program_memory, NULL);
            if (context->fence != VK_NULL_HANDLE)
                context->fn.vkDestroyFence(context->device, context->fence, NULL);
            if (context->command_pool != VK_NULL_HANDLE)
                context->fn.vkDestroyCommandPool(context->device, context->command_pool, NULL);
            if (context->descriptor_pool != VK_NULL_HANDLE)
                context->fn.vkDestroyDescriptorPool(context->device, context->descriptor_pool, NULL);
            if (context->pipeline != VK_NULL_HANDLE)
                context->fn.vkDestroyPipeline(context->device, context->pipeline, NULL);
            if (context->linear_pipeline != VK_NULL_HANDLE)
                context->fn.vkDestroyPipeline(
                    context->device, context->linear_pipeline, NULL);
            if (context->pipeline_layout != VK_NULL_HANDLE)
                context->fn.vkDestroyPipelineLayout(context->device, context->pipeline_layout, NULL);
            if (context->descriptor_layout != VK_NULL_HANDLE)
                context->fn.vkDestroyDescriptorSetLayout(
                    context->device, context->descriptor_layout, NULL);
        }
        if (context->fn.vkDestroyDevice != NULL)
            context->fn.vkDestroyDevice(context->device, NULL);
    }
    free(context);
}

static pact_result pact_vk_load_instance(
    pact_vulkan_state *state, pact_vk_functions *fn)
{
#define LOAD_INSTANCE(name) do { \
    PFN_vkVoidFunction proc = state->get_instance_proc_addr(state->instance, #name); \
    if (proc == NULL) return PACT_ERROR_BACKEND_UNAVAILABLE; \
    memcpy(&fn->name, &proc, sizeof(proc)); \
} while (0)
    LOAD_INSTANCE(vkCreateDevice);
    LOAD_INSTANCE(vkGetDeviceProcAddr);
    LOAD_INSTANCE(vkGetPhysicalDeviceMemoryProperties);
    LOAD_INSTANCE(vkEnumerateDeviceExtensionProperties);
#undef LOAD_INSTANCE
    return PACT_SUCCESS;
}

static pact_result pact_vk_load_device(pact_vulkan_context *context)
{
#define LOAD_DEVICE(name) do { \
    PFN_vkVoidFunction proc = context->fn.vkGetDeviceProcAddr(context->device, #name); \
    if (proc == NULL) return PACT_ERROR_BACKEND_UNAVAILABLE; \
    memcpy(&context->fn.name, &proc, sizeof(proc)); \
} while (0)
    LOAD_DEVICE(vkDestroyDevice); LOAD_DEVICE(vkGetDeviceQueue);
    LOAD_DEVICE(vkCreateBuffer); LOAD_DEVICE(vkDestroyBuffer);
    LOAD_DEVICE(vkGetBufferMemoryRequirements); LOAD_DEVICE(vkAllocateMemory);
    LOAD_DEVICE(vkFreeMemory); LOAD_DEVICE(vkBindBufferMemory);
    LOAD_DEVICE(vkMapMemory); LOAD_DEVICE(vkUnmapMemory);
    LOAD_DEVICE(vkCreateShaderModule); LOAD_DEVICE(vkDestroyShaderModule);
    LOAD_DEVICE(vkCreateDescriptorSetLayout); LOAD_DEVICE(vkDestroyDescriptorSetLayout);
    LOAD_DEVICE(vkCreatePipelineLayout); LOAD_DEVICE(vkDestroyPipelineLayout);
    LOAD_DEVICE(vkCreateComputePipelines); LOAD_DEVICE(vkDestroyPipeline);
    LOAD_DEVICE(vkCreateDescriptorPool); LOAD_DEVICE(vkDestroyDescriptorPool);
    LOAD_DEVICE(vkAllocateDescriptorSets); LOAD_DEVICE(vkUpdateDescriptorSets);
    LOAD_DEVICE(vkCreateCommandPool); LOAD_DEVICE(vkDestroyCommandPool);
    LOAD_DEVICE(vkResetCommandPool); LOAD_DEVICE(vkAllocateCommandBuffers);
    LOAD_DEVICE(vkBeginCommandBuffer); LOAD_DEVICE(vkEndCommandBuffer);
    LOAD_DEVICE(vkCmdCopyBuffer); LOAD_DEVICE(vkCmdPipelineBarrier);
    LOAD_DEVICE(vkCmdBindPipeline); LOAD_DEVICE(vkCmdBindDescriptorSets);
    LOAD_DEVICE(vkCmdPushConstants); LOAD_DEVICE(vkCmdDispatch);
    LOAD_DEVICE(vkCreateFence); LOAD_DEVICE(vkDestroyFence);
    LOAD_DEVICE(vkResetFences); LOAD_DEVICE(vkQueueSubmit);
    LOAD_DEVICE(vkWaitForFences); LOAD_DEVICE(vkDeviceWaitIdle);
#undef LOAD_DEVICE
    return PACT_SUCCESS;
}

static pact_result pact_vk_create_buffer(
    pact_vulkan_context *context, VkDeviceSize bytes,
    VkBufferUsageFlags usage, VkBuffer *out)
{
    VkBufferCreateInfo info = {0};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = bytes;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkResult result = context->fn.vkCreateBuffer(context->device, &info, NULL, out);
    return result == VK_SUCCESS ? PACT_SUCCESS : pact_vk_error(result);
}

static pact_result pact_vk_create_pair(
    pact_vulkan_context *context, VkDeviceSize first_size, VkDeviceSize second_size,
    VkBufferUsageFlags first_usage, VkBufferUsageFlags second_usage,
    VkMemoryPropertyFlags flags, pact_vk_pair *pair)
{
    pact_result status = pact_vk_create_buffer(
        context, first_size, first_usage, &pair->first);
    if (status != PACT_SUCCESS) return status;
    status = pact_vk_create_buffer(context, second_size, second_usage, &pair->second);
    if (status != PACT_SUCCESS) return status;
    VkMemoryRequirements first, second;
    context->fn.vkGetBufferMemoryRequirements(context->device, pair->first, &first);
    context->fn.vkGetBufferMemoryRequirements(context->device, pair->second, &second);
    uint32_t type = 0;
    if (!pact_vk_memory_type(&context->memory_properties,
                             first.memoryTypeBits & second.memoryTypeBits, flags, &type))
        return PACT_ERROR_BACKEND_UNAVAILABLE;
    if (!pact_vk_align(first.size, second.alignment, &pair->second_offset) ||
        pair->second_offset > UINT64_MAX - second.size)
        return PACT_ERROR_OUT_OF_MEMORY;
    VkMemoryAllocateInfo allocation = {0};
    allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation.allocationSize = pair->second_offset + second.size;
    allocation.memoryTypeIndex = type;
    VkResult result = context->fn.vkAllocateMemory(
        context->device, &allocation, NULL, &pair->memory);
    if (result != VK_SUCCESS) return pact_vk_error(result);
    result = context->fn.vkBindBufferMemory(context->device, pair->first, pair->memory, 0);
    if (result != VK_SUCCESS) return pact_vk_error(result);
    result = context->fn.vkBindBufferMemory(
        context->device, pair->second, pair->memory, pair->second_offset);
    if (result != VK_SUCCESS) return pact_vk_error(result);
    if (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        result = context->fn.vkMapMemory(context->device, pair->memory,
                                          0, VK_WHOLE_SIZE, 0, &pair->mapped);
        if (result != VK_SUCCESS) return pact_vk_error(result);
    }
    return PACT_SUCCESS;
}

static pact_result pact_vk_allocate_one(
    pact_vulkan_context *context, VkBuffer buffer,
    VkMemoryPropertyFlags flags, VkDeviceMemory *out_memory, void **out_mapped)
{
    VkMemoryRequirements requirements;
    context->fn.vkGetBufferMemoryRequirements(context->device, buffer, &requirements);
    uint32_t type = 0;
    if (!pact_vk_memory_type(&context->memory_properties,
            requirements.memoryTypeBits, flags, &type))
        return PACT_ERROR_BACKEND_UNAVAILABLE;
    VkMemoryAllocateInfo allocation = {0};
    allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = type;
    VkResult result = context->fn.vkAllocateMemory(
        context->device, &allocation, NULL, out_memory);
    if (result != VK_SUCCESS) return pact_vk_error(result);
    result = context->fn.vkBindBufferMemory(context->device, buffer, *out_memory, 0);
    if (result != VK_SUCCESS) return pact_vk_error(result);
    if (out_mapped != NULL) {
        result = context->fn.vkMapMemory(context->device, *out_memory,
            0, VK_WHOLE_SIZE, 0, out_mapped);
        if (result != VK_SUCCESS) return pact_vk_error(result);
    }
    return PACT_SUCCESS;
}

static pact_result pact_vk_create_program(pact_vulkan_context *context)
{
    const VkDeviceSize bytes = 64u * 4u * sizeof(uint32_t);
    pact_result status = pact_vk_create_buffer(context, bytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        &context->program_buffer);
    if (status != PACT_SUCCESS) return status;
    int integrated = context->properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
    if (integrated) {
        status = pact_vk_allocate_one(context, context->program_buffer,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            &context->program_memory, &context->program_mapped);
        if (status == PACT_SUCCESS) {
            context->program_direct = 1;
            return PACT_SUCCESS;
        }
        if (context->program_mapped != NULL)
            context->fn.vkUnmapMemory(context->device, context->program_memory);
        context->fn.vkDestroyBuffer(context->device, context->program_buffer, NULL);
        context->program_buffer = VK_NULL_HANDLE;
        if (context->program_memory != VK_NULL_HANDLE)
            context->fn.vkFreeMemory(context->device, context->program_memory, NULL);
        context->program_mapped = NULL;
        context->program_memory = VK_NULL_HANDLE;
        status = pact_vk_create_buffer(context, bytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            &context->program_buffer);
        if (status != PACT_SUCCESS) return status;
    }
    status = pact_vk_allocate_one(context, context->program_buffer,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &context->program_memory, NULL);
    if (status != PACT_SUCCESS) return status;
    status = pact_vk_create_buffer(context, bytes,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &context->program_staging_buffer);
    if (status != PACT_SUCCESS) return status;
    return pact_vk_allocate_one(context, context->program_staging_buffer,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &context->program_staging_memory, &context->program_mapped);
}

static pact_result pact_vk_create_one_pipeline(
    pact_vulkan_context *context, const uint32_t *code, size_t code_size,
    VkPipeline *out_pipeline)
{
    VkShaderModule module = VK_NULL_HANDLE;
    VkShaderModuleCreateInfo shader = {0};
    shader.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader.codeSize = code_size;
    shader.pCode = code;
    VkResult result = context->fn.vkCreateShaderModule(
        context->device, &shader, NULL, &module);
    if (result != VK_SUCCESS) return pact_vk_error(result);
    VkComputePipelineCreateInfo pipeline = {0};
    pipeline.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline.stage.module = module;
    pipeline.stage.pName = "main";
    pipeline.layout = context->pipeline_layout;
    result = context->fn.vkCreateComputePipelines(
        context->device, VK_NULL_HANDLE, 1, &pipeline, NULL, out_pipeline);
    context->fn.vkDestroyShaderModule(context->device, module, NULL);
    return result == VK_SUCCESS ? PACT_SUCCESS : pact_vk_error(result);
}

static pact_result pact_vk_create_pipeline(pact_vulkan_context *context)
{
    VkDescriptorSetLayoutBinding bindings[3] = {0};
    for (uint32_t i = 0; i < 3; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layout = {0};
    layout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout.bindingCount = 3;
    layout.pBindings = bindings;
    VkResult result = context->fn.vkCreateDescriptorSetLayout(
        context->device, &layout, NULL, &context->descriptor_layout);
    if (result != VK_SUCCESS) return pact_vk_error(result);
    VkPushConstantRange push = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 2u * sizeof(uint32_t)};
    VkPipelineLayoutCreateInfo pipeline_layout = {0};
    pipeline_layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout.setLayoutCount = 1;
    pipeline_layout.pSetLayouts = &context->descriptor_layout;
    pipeline_layout.pushConstantRangeCount = 1;
    pipeline_layout.pPushConstantRanges = &push;
    result = context->fn.vkCreatePipelineLayout(
        context->device, &pipeline_layout, NULL, &context->pipeline_layout);
    if (result != VK_SUCCESS) return pact_vk_error(result);
    pact_result status = pact_vk_create_one_pipeline(context,
        pact_graph_f32_spv, sizeof(pact_graph_f32_spv), &context->pipeline);
    if (status != PACT_SUCCESS) return status;
    status = pact_vk_create_one_pipeline(context,
        pact_linear_f32_spv, sizeof(pact_linear_f32_spv), &context->linear_pipeline);
    if (status != PACT_SUCCESS) return status;
    VkDescriptorPoolSize pool_size = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    VkDescriptorPoolCreateInfo pool = {0};
    pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool.maxSets = 1;
    pool.poolSizeCount = 1;
    pool.pPoolSizes = &pool_size;
    result = context->fn.vkCreateDescriptorPool(
        context->device, &pool, NULL, &context->descriptor_pool);
    if (result != VK_SUCCESS) return pact_vk_error(result);
    VkDescriptorSetAllocateInfo set_info = {0};
    set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    set_info.descriptorPool = context->descriptor_pool;
    set_info.descriptorSetCount = 1;
    set_info.pSetLayouts = &context->descriptor_layout;
    result = context->fn.vkAllocateDescriptorSets(
        context->device, &set_info, &context->descriptor_set);
    return result == VK_SUCCESS ? PACT_SUCCESS : pact_vk_error(result);
}

static pact_result pact_vk_create_commands(pact_vulkan_context *context, uint32_t family)
{
    VkCommandPoolCreateInfo pool = {0};
    pool.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool.queueFamilyIndex = family;
    pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkResult result = context->fn.vkCreateCommandPool(
        context->device, &pool, NULL, &context->command_pool);
    if (result != VK_SUCCESS) return pact_vk_error(result);
    VkCommandBufferAllocateInfo cmd = {0};
    cmd.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd.commandPool = context->command_pool;
    cmd.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd.commandBufferCount = 1;
    result = context->fn.vkAllocateCommandBuffers(
        context->device, &cmd, &context->command_buffer);
    if (result != VK_SUCCESS) return pact_vk_error(result);
    VkFenceCreateInfo fence = {0};
    fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    result = context->fn.vkCreateFence(context->device, &fence, NULL, &context->fence);
    return result == VK_SUCCESS ? PACT_SUCCESS : pact_vk_error(result);
}

static pact_result pact_vk_create_context(
    pact_runtime *runtime, uint32_t device_index, pact_vulkan_context **out)
{
    pact_vulkan_state *state = runtime->vulkan_state;
    VkPhysicalDevice physical = (VkPhysicalDevice)runtime->devices[device_index].native_device;
    pact_vulkan_context *context = calloc(1, sizeof(*context));
    if (context == NULL) return PACT_ERROR_OUT_OF_MEMORY;
    context->physical = physical;
    state->get_physical_device_properties(physical, &context->properties);
    pact_result status = pact_vk_load_instance(state, &context->fn);
    if (status != PACT_SUCCESS) goto fail;
    uint32_t family_count = 0;
    state->get_queue_family_properties(physical, &family_count, NULL);
    if (family_count == 0) { status = PACT_ERROR_BACKEND_UNAVAILABLE; goto fail; }
    VkQueueFamilyProperties *families = calloc(family_count, sizeof(*families));
    if (families == NULL) { status = PACT_ERROR_OUT_OF_MEMORY; goto fail; }
    state->get_queue_family_properties(physical, &family_count, families);
    uint32_t family = UINT32_MAX;
    for (uint32_t i = 0; i < family_count; ++i)
        if (families[i].queueCount && (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
            family = i;
            break;
        }
    free(families);
    if (family == UINT32_MAX || context->properties.limits.maxComputeWorkGroupInvocations < 256 ||
        context->properties.limits.maxComputeWorkGroupSize[0] < 256) {
        status = PACT_ERROR_BACKEND_UNAVAILABLE;
        goto fail;
    }
    uint32_t extension_count = 0;
    VkResult result = context->fn.vkEnumerateDeviceExtensionProperties(
        physical, NULL, &extension_count, NULL);
    if (result != VK_SUCCESS) { status = pact_vk_error(result); goto fail; }
    VkExtensionProperties *extensions = NULL;
    if (extension_count != 0) {
        extensions = calloc(extension_count, sizeof(*extensions));
        if (extensions == NULL) { status = PACT_ERROR_OUT_OF_MEMORY; goto fail; }
        result = context->fn.vkEnumerateDeviceExtensionProperties(
            physical, NULL, &extension_count, extensions);
        if (result != VK_SUCCESS) {
            free(extensions);
            status = pact_vk_error(result);
            goto fail;
        }
    }
    const char *portability = "VK_KHR_portability_subset";
    int has_portability = 0;
    for (uint32_t i = 0; i < extension_count; ++i)
        if (strcmp(extensions[i].extensionName, portability) == 0)
            has_portability = 1;
    free(extensions);
    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue = {0};
    queue.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue.queueFamilyIndex = family;
    queue.queueCount = 1;
    queue.pQueuePriorities = &priority;
    VkDeviceCreateInfo device = {0};
    device.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device.queueCreateInfoCount = 1;
    device.pQueueCreateInfos = &queue;
    if (has_portability) {
        device.enabledExtensionCount = 1;
        device.ppEnabledExtensionNames = &portability;
    }
    result = context->fn.vkCreateDevice(physical, &device, NULL, &context->device);
    if (result != VK_SUCCESS) { status = pact_vk_error(result); goto fail; }
    status = pact_vk_load_device(context);
    if (status != PACT_SUCCESS) goto fail;
    context->fn.vkGetDeviceQueue(context->device, family, 0, &context->queue);
    context->fn.vkGetPhysicalDeviceMemoryProperties(physical, &context->memory_properties);
    status = pact_vk_create_program(context);
    if (status != PACT_SUCCESS) goto fail;
    status = pact_vk_create_pipeline(context);
    if (status != PACT_SUCCESS) goto fail;
    status = pact_vk_create_commands(context, family);
    if (status != PACT_SUCCESS) goto fail;
    *out = context;
    return PACT_SUCCESS;
fail:
    pact_vulkan_context_destroy(context);
    return status;
}

static pact_result pact_vk_workspace(
    pact_vulkan_context *context, size_t sources, size_t elements)
{
    if (context->capacity >= elements && context->source_capacity >= sources)
        return PACT_SUCCESS;
    pact_vk_pair_destroy(context, &context->device_data);
    pact_vk_pair_destroy(context, &context->staging);
    context->capacity = 0;
    context->source_capacity = 0;
    context->direct_access = 0;
    if (sources == 0 || elements > SIZE_MAX / (sources * sizeof(float)))
        return PACT_ERROR_OUT_OF_MEMORY;
    VkDeviceSize input_bytes = sources * elements * sizeof(float);
    VkDeviceSize output_bytes = elements * sizeof(float);
    VkBufferUsageFlags input_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBufferUsageFlags output_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    int integrated = context->properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
    pact_result status = PACT_ERROR_BACKEND_UNAVAILABLE;
    if (integrated) {
        status = pact_vk_create_pair(context, input_bytes, output_bytes,
            input_usage, output_usage,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
            VK_MEMORY_PROPERTY_HOST_CACHED_BIT, &context->device_data);
        if (status == PACT_SUCCESS) context->direct_access = 1;
        else pact_vk_pair_destroy(context, &context->device_data);
    }
    if (!context->direct_access) {
        status = pact_vk_create_pair(context, input_bytes, output_bytes,
            input_usage, output_usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            &context->device_data);
        if (status != PACT_SUCCESS) return status;
        status = pact_vk_create_pair(context, input_bytes, output_bytes,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
            VK_MEMORY_PROPERTY_HOST_CACHED_BIT, &context->staging);
        if (status == PACT_ERROR_BACKEND_UNAVAILABLE) {
            pact_vk_pair_destroy(context, &context->staging);
            status = pact_vk_create_pair(context, input_bytes, output_bytes,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &context->staging);
        }
        if (status != PACT_SUCCESS) return status;
    }
    VkDescriptorBufferInfo buffers[3] = {0};
    buffers[0].buffer = context->device_data.first;
    buffers[0].range = input_bytes;
    buffers[1].buffer = context->program_buffer;
    buffers[1].range = 64u * 4u * sizeof(uint32_t);
    buffers[2].buffer = context->device_data.second;
    buffers[2].range = output_bytes;
    VkWriteDescriptorSet writes[3] = {0};
    for (uint32_t i = 0; i < 3; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = context->descriptor_set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buffers[i];
    }
    context->fn.vkUpdateDescriptorSets(context->device, 3, writes, 0, NULL);
    context->capacity = elements;
    context->source_capacity = sources;
    return PACT_SUCCESS;
}

static void pact_vk_barrier(
    pact_vulkan_context *context, VkPipelineStageFlags source_stage,
    VkPipelineStageFlags target_stage, VkAccessFlags source_access,
    VkAccessFlags target_access)
{
    VkMemoryBarrier barrier = {0};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = source_access;
    barrier.dstAccessMask = target_access;
    context->fn.vkCmdPipelineBarrier(context->command_buffer,
        source_stage, target_stage, 0, 1, &barrier, 0, NULL, 0, NULL);
}

static uint32_t pact_vk_linear_program(
    const pact_graph *graph, const pact_task *tasks,
    const uint32_t *source_slots, uint32_t *instructions)
{
    uint32_t slots[64] = {0};
    uint32_t reverse[64] = {0};
    size_t steps = 0;
    size_t current = graph->count - 1;
    while (tasks[current].kind == PACT_TASK_ADD) {
        size_t a = tasks[current].dependencies[0];
        size_t b = tasks[current].dependencies[1];
        if (tasks[b].kind == PACT_TASK_SOURCE) {
            slots[steps] = source_slots[b];
            reverse[steps++] = 0;
            current = a;
        } else if (tasks[a].kind == PACT_TASK_SOURCE) {
            slots[steps] = source_slots[a];
            reverse[steps++] = 1;
            current = b;
        } else return 0;
    }
    if (steps == 0 || tasks[current].kind != PACT_TASK_SOURCE)
        return 0;
    instructions[0] = source_slots[current];
    instructions[1] = 0;
    instructions[2] = 0;
    instructions[3] = 0;
    for (size_t i = 0; i < steps; ++i) {
        size_t from = steps - 1 - i;
        instructions[(i + 1) * 4] = slots[from];
        instructions[(i + 1) * 4 + 1] = reverse[from];
        instructions[(i + 1) * 4 + 2] = 0;
        instructions[(i + 1) * 4 + 3] = 0;
    }
    return (uint32_t)steps;
}

static pact_result pact_vk_run_chunk(
    pact_vulkan_context *context, const pact_graph *graph,
    const pact_task *tasks, size_t source_count, size_t start, size_t count,
    unsigned char *output, pact_gpu_timing *timing)
{
    double upload_begin = pact_now_us();
    size_t source_slot = 0;
    size_t input_bytes = count * source_count * sizeof(float);
    unsigned char *input = context->direct_access
        ? context->device_data.mapped : context->staging.mapped;
    uint32_t *instructions = context->program_mapped;
    uint32_t source_slots[64] = {0};
    for (size_t node = 0; node < graph->count; ++node) {
        if (tasks[node].kind == PACT_TASK_SOURCE) {
            pact_tensor *source = graph->nodes[node];
            memcpy(input + source_slot * count * sizeof(float),
                   source->host_data + start * sizeof(float), count * sizeof(float));
            source_slots[node] = (uint32_t)source_slot;
            instructions[node * 4] = 0;
            instructions[node * 4 + 1] = (uint32_t)source_slot++;
            instructions[node * 4 + 2] = 0;
            instructions[node * 4 + 3] = 0;
        } else {
            instructions[node * 4] = 1;
            instructions[node * 4 + 1] = (uint32_t)tasks[node].dependencies[0];
            instructions[node * 4 + 2] = (uint32_t)tasks[node].dependencies[1];
            instructions[node * 4 + 3] = 0;
        }
    }
    uint32_t linear_steps = pact_vk_linear_program(
        graph, tasks, source_slots, instructions);
    timing->upload_us += pact_now_us() - upload_begin;
    double record_begin = pact_now_us();
    VkResult result = context->fn.vkResetCommandPool(context->device,
                                                       context->command_pool, 0);
    if (result != VK_SUCCESS) return pact_vk_error(result);
    VkCommandBufferBeginInfo begin = {0};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    result = context->fn.vkBeginCommandBuffer(context->command_buffer, &begin);
    if (result != VK_SUCCESS) return pact_vk_error(result);
    if (!context->direct_access) {
        VkBufferCopy copy = {0};
        copy.size = input_bytes;
        context->fn.vkCmdCopyBuffer(context->command_buffer,
            context->staging.first, context->device_data.first, 1, &copy);
    }
    if (!context->program_direct) {
        VkBufferCopy copy = {0};
        copy.size = 64u * 4u * sizeof(uint32_t);
        context->fn.vkCmdCopyBuffer(context->command_buffer,
            context->program_staging_buffer, context->program_buffer, 1, &copy);
    }
    if (!context->direct_access || !context->program_direct)
        pact_vk_barrier(context, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    context->fn.vkCmdBindPipeline(context->command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        linear_steps != 0 ? context->linear_pipeline : context->pipeline);
    context->fn.vkCmdBindDescriptorSets(context->command_buffer,
        VK_PIPELINE_BIND_POINT_COMPUTE, context->pipeline_layout,
        0, 1, &context->descriptor_set, 0, NULL);
    uint32_t parameters[2] = {(uint32_t)count,
                              linear_steps != 0 ? linear_steps : (uint32_t)graph->count};
    context->fn.vkCmdPushConstants(context->command_buffer,
        context->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(parameters), parameters);
    context->fn.vkCmdDispatch(context->command_buffer,
        ((uint32_t)count + 255u) / 256u, 1, 1);
    if (!context->direct_access) {
        pact_vk_barrier(context, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        VkBufferCopy copy = {0};
        copy.size = count * sizeof(float);
        context->fn.vkCmdCopyBuffer(context->command_buffer,
            context->device_data.second, context->staging.second, 1, &copy);
    }
    result = context->fn.vkEndCommandBuffer(context->command_buffer);
    if (result != VK_SUCCESS) return pact_vk_error(result);
    timing->record_us += pact_now_us() - record_begin;
    double queue_begin = pact_now_us();
    VkSubmitInfo submit = {0};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &context->command_buffer;
    result = context->fn.vkQueueSubmit(context->queue, 1, &submit, context->fence);
    if (result != VK_SUCCESS) return pact_vk_error(result);
    result = context->fn.vkWaitForFences(context->device, 1, &context->fence,
                                           VK_TRUE, UINT64_MAX);
    if (result != VK_SUCCESS) return PACT_ERROR_EXECUTION;
    timing->queue_us += pact_now_us() - queue_begin;
    double download_begin = pact_now_us();
    unsigned char *result_data = context->direct_access
        ? (unsigned char *)context->device_data.mapped + context->device_data.second_offset
        : (unsigned char *)context->staging.mapped + context->staging.second_offset;
    memcpy(output + start * sizeof(float), result_data, count * sizeof(float));
    timing->download_us += pact_now_us() - download_begin;
    result = context->fn.vkResetFences(context->device, 1, &context->fence);
    return result == VK_SUCCESS ? PACT_SUCCESS : pact_vk_error(result);
}

pact_result pact_vulkan_execute_tasks(
    pact_runtime *runtime, uint32_t device_index, const pact_graph *graph,
    const pact_task *tasks, size_t task_count, unsigned char *output,
    pact_gpu_timing *out_timing)
{
    if (out_timing != NULL) *out_timing = (pact_gpu_timing){0};
    if (runtime == NULL || device_index >= runtime->device_count ||
        runtime->devices[device_index].info.backend != PACT_BACKEND_VULKAN ||
        graph == NULL || graph->count == 0 || graph->count > 64 ||
        tasks == NULL || task_count != graph->count + 1 ||
        tasks[task_count - 1].kind != PACT_TASK_DOWNLOAD)
        return PACT_ERROR_INVALID_ARGUMENT;
    size_t source_count = 0;
    for (size_t i = 0; i < graph->count; ++i) {
        if (tasks[i].node_index != i || tasks[i].value_id != graph->nodes[i]->value_id ||
            tasks[i].version != graph->nodes[i]->version ||
            tasks[i].start != tasks[0].start || tasks[i].count != tasks[0].count ||
            tasks[i].device_index != device_index)
            return PACT_ERROR_INVALID_ARGUMENT;
        if (tasks[i].kind == PACT_TASK_SOURCE) ++source_count;
        else if (tasks[i].kind != PACT_TASK_ADD ||
                 tasks[i].dependencies[0] >= i || tasks[i].dependencies[1] >= i)
            return PACT_ERROR_INVALID_ARGUMENT;
    }
    if (source_count == 0 || tasks[0].count == 0 ||
        tasks[0].start > graph->nodes[graph->count - 1]->element_count ||
        tasks[0].count > graph->nodes[graph->count - 1]->element_count - tasks[0].start)
        return PACT_ERROR_INVALID_ARGUMENT;
    pact_gpu_timing timing = {0};
    double setup_begin = pact_now_us();
    pact_device_record *record = &runtime->devices[device_index];
    if (record->execution_status != PACT_SUCCESS) return record->execution_status;
    pact_vulkan_context *context = record->execution_context;
    if (context == NULL) {
        pact_result status = pact_vk_create_context(runtime, device_index, &context);
        if (status != PACT_SUCCESS) {
            record->execution_status = status;
            return status;
        }
        record->execution_context = context;
    }
    if (context->failed) return PACT_ERROR_BACKEND_UNAVAILABLE;
    const size_t budget = 64u * 1024u * 1024u;
    size_t bytes_per_element = (source_count + 1) * sizeof(float) *
                               (context->direct_access ? 1u : 2u);
    size_t chunk = budget / bytes_per_element;
    size_t max_range = context->properties.limits.maxStorageBufferRange / sizeof(float);
    if (chunk > max_range / source_count) chunk = max_range / source_count;
    if (chunk > max_range) chunk = max_range;
    size_t max_groups = context->properties.limits.maxComputeWorkGroupCount[0];
    size_t max_dispatch = max_groups > SIZE_MAX / 256u
        ? SIZE_MAX : max_groups * 256u;
    if (chunk > max_dispatch) chunk = max_dispatch;
    if (chunk > 1048576u) chunk = 1048576u;
    size_t maximum_chunk = chunk;
    if (chunk > tasks[0].count) chunk = tasks[0].count;
    if (chunk == 0) return PACT_ERROR_BACKEND_UNAVAILABLE;
    size_t reserve = 1;
    while (reserve < chunk && reserve <= maximum_chunk / 2) reserve *= 2;
    if (reserve < chunk) reserve = maximum_chunk;
    pact_result status = PACT_ERROR_OUT_OF_MEMORY;
    while (reserve != 0) {
        status = pact_vk_workspace(context, source_count, reserve);
        if (status == PACT_SUCCESS) break;
        if (status != PACT_ERROR_OUT_OF_MEMORY) return status;
        reserve /= 2;
    }
    if (status != PACT_SUCCESS) return status;
    if (chunk > reserve) chunk = reserve;
    timing.setup_us = pact_now_us() - setup_begin;
    double begin = pact_now_us();
    size_t end = tasks[0].start + tasks[0].count;
    for (size_t start = tasks[0].start; start < end;) {
        size_t count = end - start;
        if (count > chunk) count = chunk;
        status = pact_vk_run_chunk(context, graph, tasks, source_count,
                                    start, count, output, &timing);
        if (status != PACT_SUCCESS) {
            context->failed = 1;
            record->execution_status = status;
            break;
        }
        start += count;
    }
    timing.active_us = pact_now_us() - begin;
    if (out_timing != NULL) *out_timing = timing;
    return status;
}
