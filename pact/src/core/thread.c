#include "thread.h"

#include <stdlib.h>
#include <stdint.h>

#if defined(_WIN32)
#include <process.h>
#endif

typedef struct pact_thread_launch {
    pact_thread_fn fn;
    void *argument;
} pact_thread_launch;

#if defined(_WIN32)
static unsigned __stdcall pact_thread_entry(void *opaque)
#else
static void *pact_thread_entry(void *opaque)
#endif
{
    pact_thread_launch launch = *(pact_thread_launch *)opaque;
    free(opaque);
    int result = launch.fn(launch.argument);
#if defined(_WIN32)
    return (unsigned)result;
#else
    return (void *)(size_t)result;
#endif
}

int pact_thread_start(pact_thread *thread, pact_thread_fn fn, void *argument)
{
    pact_thread_launch *launch = malloc(sizeof(*launch));
    if (launch == NULL) return 0;
    launch->fn = fn;
    launch->argument = argument;
#if defined(_WIN32)
    uintptr_t handle = _beginthreadex(NULL, 0, pact_thread_entry, launch, 0, NULL);
    if (handle == 0) { free(launch); return 0; }
    *thread = (HANDLE)handle;
#else
    if (pthread_create(thread, NULL, pact_thread_entry, launch) != 0) {
        free(launch);
        return 0;
    }
#endif
    return 1;
}

void pact_thread_join(pact_thread thread)
{
#if defined(_WIN32)
    WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
#else
    pthread_join(thread, NULL);
#endif
}

int pact_mutex_init(pact_mutex *mutex)
{
#if defined(_WIN32)
    InitializeCriticalSection(mutex);
    return 1;
#else
    return pthread_mutex_init(mutex, NULL) == 0;
#endif
}

void pact_mutex_destroy(pact_mutex *mutex)
{
#if defined(_WIN32)
    DeleteCriticalSection(mutex);
#else
    pthread_mutex_destroy(mutex);
#endif
}

void pact_mutex_lock(pact_mutex *mutex)
{
#if defined(_WIN32)
    EnterCriticalSection(mutex);
#else
    pthread_mutex_lock(mutex);
#endif
}

void pact_mutex_unlock(pact_mutex *mutex)
{
#if defined(_WIN32)
    LeaveCriticalSection(mutex);
#else
    pthread_mutex_unlock(mutex);
#endif
}

int pact_condition_init(pact_condition *condition)
{
#if defined(_WIN32)
    InitializeConditionVariable(condition);
    return 1;
#else
    return pthread_cond_init(condition, NULL) == 0;
#endif
}

void pact_condition_destroy(pact_condition *condition)
{
#if !defined(_WIN32)
    pthread_cond_destroy(condition);
#else
    (void)condition;
#endif
}

void pact_condition_wait(pact_condition *condition, pact_mutex *mutex)
{
#if defined(_WIN32)
    SleepConditionVariableCS(condition, mutex, INFINITE);
#else
    pthread_cond_wait(condition, mutex);
#endif
}

void pact_condition_signal(pact_condition *condition)
{
#if defined(_WIN32)
    WakeConditionVariable(condition);
#else
    pthread_cond_signal(condition);
#endif
}

void pact_condition_broadcast(pact_condition *condition)
{
#if defined(_WIN32)
    WakeAllConditionVariable(condition);
#else
    pthread_cond_broadcast(condition);
#endif
}
