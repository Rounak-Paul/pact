#include "runtime_internal.h"

#include <string.h>

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <unistd.h>
#endif

static uint32_t pact_cpu_logical_processor_count(void)
{
#if defined(_WIN32)
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    return info.dwNumberOfProcessors == 0 ? 1 : info.dwNumberOfProcessors;
#elif defined(_SC_NPROCESSORS_ONLN)
    long count = sysconf(_SC_NPROCESSORS_ONLN);
    return count > 0 && (unsigned long)count <= UINT32_MAX ? (uint32_t)count : 1;
#else
    return 1;
#endif
}

pact_result pact_cpu_discover(pact_runtime *runtime)
{
    pact_device_info info;
    memset(&info, 0, sizeof(info));
    info.struct_size = sizeof(info);
    info.backend = PACT_BACKEND_CPU;
    info.type = PACT_DEVICE_TYPE_CPU;
    info.logical_processor_count = pact_cpu_logical_processor_count();
    memcpy(info.name, "Host CPU", sizeof("Host CPU"));
    return pact_runtime_add_device(runtime, &info, NULL);
}
