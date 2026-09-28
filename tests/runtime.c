#include <pact/pact.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition)                                                           \
    do {                                                                           \
        if (!(condition)) {                                                        \
            fprintf(stderr, "check failed: %s at line %d\n", #condition, __LINE__); \
            return 1;                                                              \
        }                                                                          \
    } while (0)

int main(void)
{
    pact_runtime *runtime = NULL;
    CHECK(pact_runtime_create(NULL) == PACT_ERROR_INVALID_ARGUMENT);
    CHECK(pact_runtime_create(&runtime) == PACT_SUCCESS);
    CHECK(runtime != NULL);

    uint32_t count = 0;
    CHECK(pact_runtime_get_device_count(runtime, &count) == PACT_SUCCESS);
    CHECK(count >= 1);

    int found_cpu = 0;
    int found_vulkan = 0;
    for (uint32_t i = 0; i < count; ++i) {
        pact_device_info info = {0};
        info.struct_size = sizeof(info);
        CHECK(pact_runtime_get_device_info(runtime, i, &info) == PACT_SUCCESS);
        CHECK(info.name[0] != '\0');
        CHECK(info.struct_size == sizeof(info));
        if (info.backend == PACT_BACKEND_CPU) {
            found_cpu = 1;
            CHECK(info.type == PACT_DEVICE_TYPE_CPU);
            CHECK(info.logical_processor_count >= 1);
        }
        if (info.backend == PACT_BACKEND_VULKAN) {
            found_vulkan = 1;
            CHECK(info.type != PACT_DEVICE_TYPE_CPU);
        }
        printf("%u: %s (backend %d)\n", i, info.name, (int)info.backend);
    }
    CHECK(found_cpu);
    CHECK(pact_runtime_get_device_info(runtime, count, NULL) ==
          PACT_ERROR_INVALID_ARGUMENT);
    pact_device_info invalid_info = {0};
    CHECK(pact_runtime_get_device_info(runtime, 0, &invalid_info) ==
          PACT_ERROR_INVALID_ARGUMENT);

    pact_result backend_status = PACT_ERROR_INTERNAL;
    CHECK(pact_runtime_get_backend_status(runtime, PACT_BACKEND_CPU, &backend_status) ==
          PACT_SUCCESS);
    CHECK(backend_status == PACT_SUCCESS);
    CHECK(pact_runtime_get_backend_status(runtime, PACT_BACKEND_VULKAN, &backend_status) ==
          PACT_SUCCESS);
    CHECK((backend_status == PACT_SUCCESS) == found_vulkan);
    if (getenv("PACT_TEST_EXPECT_NO_VULKAN") != NULL) {
        CHECK(!found_vulkan);
        CHECK(backend_status == PACT_ERROR_BACKEND_UNAVAILABLE);
    }
    CHECK(pact_runtime_get_backend_status(runtime, (pact_backend)0, &backend_status) ==
          PACT_ERROR_INVALID_ARGUMENT);

    CHECK(strcmp(pact_result_string(PACT_SUCCESS), "success") == 0);
    pact_runtime_destroy(runtime);
    pact_runtime_destroy(NULL);
    return 0;
}
