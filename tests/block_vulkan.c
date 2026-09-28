#define _GNU_SOURCE
#include <dlfcn.h>
#include <string.h>

void *dlopen(const char *filename, int flags)
{
    if (filename != NULL && strstr(filename, "libvulkan") != NULL) {
        return NULL;
    }
    void *(*real_dlopen)(const char *, int) = dlsym(RTLD_NEXT, "dlopen");
    return real_dlopen(filename, flags);
}
