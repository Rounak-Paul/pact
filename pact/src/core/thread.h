#ifndef PACT_THREAD_H
#define PACT_THREAD_H

#if defined(_WIN32)
#include <windows.h>
typedef HANDLE pact_thread;
typedef CRITICAL_SECTION pact_mutex;
typedef CONDITION_VARIABLE pact_condition;
#else
#include <pthread.h>
typedef pthread_t pact_thread;
typedef pthread_mutex_t pact_mutex;
typedef pthread_cond_t pact_condition;
#endif

typedef int (*pact_thread_fn)(void *);

int pact_thread_start(pact_thread *thread, pact_thread_fn fn, void *argument);
void pact_thread_join(pact_thread thread);
int pact_mutex_init(pact_mutex *mutex);
void pact_mutex_destroy(pact_mutex *mutex);
void pact_mutex_lock(pact_mutex *mutex);
void pact_mutex_unlock(pact_mutex *mutex);
int pact_condition_init(pact_condition *condition);
void pact_condition_destroy(pact_condition *condition);
void pact_condition_wait(pact_condition *condition, pact_mutex *mutex);
void pact_condition_signal(pact_condition *condition);
void pact_condition_broadcast(pact_condition *condition);

#endif
