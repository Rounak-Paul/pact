#if !defined(_WIN32)
#define _POSIX_C_SOURCE 200809L
#endif
#include <pact/pact.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static void timestamp(struct timespec *out)
{
#if defined(_WIN32)
    timespec_get(out, TIME_UTC);
#else
    clock_gettime(CLOCK_MONOTONIC, out);
#endif
}

static double elapsed_us(struct timespec begin, struct timespec end)
{
    return (double)(end.tv_sec - begin.tv_sec) * 1000000.0 +
           (double)(end.tv_nsec - begin.tv_nsec) / 1000.0;
}

static void require(pact_result result, const char *operation)
{
    if (result != PACT_SUCCESS) {
        fprintf(stderr, "%s: %s\n", operation, pact_result_string(result));
        exit(1);
    }
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "usage: %s elements additions samples\n", argv[0]);
        return 2;
    }
    size_t elements = (size_t)strtoull(argv[1], NULL, 10);
    unsigned additions = (unsigned)strtoul(argv[2], NULL, 10);
    unsigned samples = (unsigned)strtoul(argv[3], NULL, 10);
    if (elements == 0 || additions == 0 || samples == 0 ||
        elements > SIZE_MAX / sizeof(float)) return 2;
    size_t bytes = elements * sizeof(float);
    float *a = malloc(bytes);
    float *b = malloc(bytes);
    float *out = malloc(bytes);
    if (a == NULL || b == NULL || out == NULL) return 1;
    for (size_t i = 0; i < elements; ++i) {
        a[i] = (float)(i % 1024);
        b[i] = (float)(i % 16);
    }
    pact_runtime *runtime = NULL;
    require(pact_runtime_create(&runtime), "runtime create");
    uint64_t shape = elements;
    puts("sample,read_us,cpu_elements,vulkan_elements,vulkan_shards,fallback,"
         "cpu_tasks,cpu_us,vulkan_active_us,vulkan_setup_us,"
         "vulkan_upload_us,vulkan_record_us,vulkan_queue_us,vulkan_download_us");
    for (unsigned sample = 0; sample < samples + 3; ++sample) {
        pact_tensor *x = NULL;
        pact_tensor *y = NULL;
        pact_tensor *current = NULL;
        require(pact_tensor_from_data(runtime, PACT_DTYPE_F32, &shape, 1,
                                      a, bytes, &x), "input a");
        require(pact_tensor_from_data(runtime, PACT_DTYPE_F32, &shape, 1,
                                      b, bytes, &y), "input b");
        current = x;
        pact_tensor_retain(current);
        for (unsigned op = 0; op < additions; ++op) {
            pact_tensor *next = NULL;
            require(pact_tensor_add(current, y, &next), "add");
            pact_tensor_release(current);
            current = next;
        }
        struct timespec begin, end;
        timestamp(&begin);
        require(pact_tensor_read(current, out, bytes), "read");
        timestamp(&end);
        for (size_t i = 0; i < elements; i += elements / 17 + 1) {
            if (out[i] != a[i] + (float)additions * b[i]) {
                fprintf(stderr, "wrong result at %zu\n", i);
                return 1;
            }
        }
        pact_execution_info info = {0};
        info.struct_size = sizeof(info);
        require(pact_runtime_get_last_execution(runtime, &info), "execution info");
        {
            printf("%d,%.3f,%llu,%llu,%u,%d,%u,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n", (int)sample - 3,
                   elapsed_us(begin, end),
                   (unsigned long long)info.cpu_elements,
                   (unsigned long long)info.vulkan_elements,
                   info.vulkan_shards, (int)info.vulkan_fallback,
                   info.cpu_tasks, info.cpu_time_us, info.vulkan_active_us,
                   info.vulkan_setup_us, info.vulkan_upload_us,
                   info.vulkan_record_us, info.vulkan_queue_us,
                   info.vulkan_download_us);
        }
        pact_tensor_release(current);
        pact_tensor_release(x);
        pact_tensor_release(y);
    }
    pact_runtime_destroy(runtime);
    free(a);
    free(b);
    free(out);
    return 0;
}
