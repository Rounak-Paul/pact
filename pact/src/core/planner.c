#include "tensor_internal.h"
#include "cpu_pool.h"
#include "thread.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct pact_linear_step {
    const float *operand;
    int operand_first;
} pact_linear_step;

typedef struct pact_linear_program {
    const float *base;
    pact_linear_step *steps;
    size_t count;
} pact_linear_program;

typedef struct pact_cpu_work {
    const pact_graph *graph;
    const pact_task *tasks;
    const pact_linear_program *linear;
    size_t task_count;
    size_t start;
    size_t count;
    unsigned char *output;
    pact_result status;
} pact_cpu_work;

typedef struct pact_gpu_work {
    pact_runtime *runtime;
    const pact_graph *graph;
    pact_task *tasks;
    size_t task_count;
    size_t start;
    size_t count;
    uint32_t device_index;
    unsigned char *output;
    pact_gpu_timing timing;
    pact_result status;
    pact_thread thread;
    int started;
} pact_gpu_work;

static double pact_now_us(void)
{
    struct timespec time;
    timespec_get(&time, TIME_UTC);
    return (double)time.tv_sec * 1000000.0 + (double)time.tv_nsec / 1000.0;
}

static pact_result pact_make_tasks(
    const pact_graph *graph, size_t start, size_t count, uint32_t device_index,
    int download, pact_task **out_tasks, size_t *out_count)
{
    size_t total = graph->count + (download ? 1u : 0u);
    if (total < graph->count || total > SIZE_MAX / sizeof(pact_task))
        return PACT_ERROR_OUT_OF_MEMORY;
    pact_task *tasks = calloc(total, sizeof(*tasks));
    if (tasks == NULL) return PACT_ERROR_OUT_OF_MEMORY;
    for (size_t i = 0; i < graph->count; ++i) {
        pact_tensor *node = graph->nodes[i];
        tasks[i].kind = node->materialized ? PACT_TASK_SOURCE : PACT_TASK_ADD;
        tasks[i].node_index = i;
        tasks[i].value_id = node->value_id;
        tasks[i].version = node->version;
        tasks[i].start = start;
        tasks[i].count = count;
        tasks[i].device_index = device_index;
        if (!node->materialized) {
            if (node->op != PACT_OP_ADD || node->input_a == NULL ||
                node->input_b == NULL) {
                free(tasks);
                return PACT_ERROR_INTERNAL;
            }
            tasks[i].dependencies[0] = node->input_a->plan_index;
            tasks[i].dependencies[1] = node->input_b->plan_index;
            if (tasks[i].dependencies[0] >= i || tasks[i].dependencies[1] >= i) {
                free(tasks);
                return PACT_ERROR_INTERNAL;
            }
        }
    }
    if (download) {
        tasks[graph->count].kind = PACT_TASK_DOWNLOAD;
        tasks[graph->count].dependencies[0] = graph->count - 1;
        tasks[graph->count].start = start;
        tasks[graph->count].count = count;
        tasks[graph->count].device_index = device_index;
    }
    *out_tasks = tasks;
    *out_count = total;
    return PACT_SUCCESS;
}

static pact_result pact_linear_compile(
    const pact_graph *graph, pact_linear_program *program)
{
    *program = (pact_linear_program){0};
    if (graph->count < 3) return PACT_ERROR_BACKEND_UNAVAILABLE;
    program->steps = calloc(graph->count, sizeof(*program->steps));
    if (program->steps == NULL) return PACT_ERROR_OUT_OF_MEMORY;
    pact_tensor *node = graph->nodes[graph->count - 1];
    while (!node->materialized && node->op == PACT_OP_ADD) {
        pact_tensor *a = node->input_a;
        pact_tensor *b = node->input_b;
        if (a == NULL || b == NULL) break;
        pact_linear_step step = {0};
        if (b->materialized) {
            step.operand = (const float *)b->host_data;
            step.operand_first = 0;
            node = a;
        } else if (a->materialized) {
            step.operand = (const float *)a->host_data;
            step.operand_first = 1;
            node = b;
        } else break;
        program->steps[program->count++] = step;
    }
    if (!node->materialized || program->count == 0) {
        free(program->steps);
        *program = (pact_linear_program){0};
        return PACT_ERROR_BACKEND_UNAVAILABLE;
    }
    program->base = (const float *)node->host_data;
    for (size_t i = 0; i < program->count / 2; ++i) {
        pact_linear_step swap = program->steps[i];
        program->steps[i] = program->steps[program->count - 1 - i];
        program->steps[program->count - 1 - i] = swap;
    }
    return PACT_SUCCESS;
}

static pact_result pact_cpu_generic(pact_cpu_work *work)
{
    const pact_graph *graph = work->graph;
    float **values = calloc(graph->count, sizeof(*values));
    size_t *remaining = calloc(graph->count, sizeof(*remaining));
    if (values == NULL || remaining == NULL) {
        free(values); free(remaining);
        return PACT_ERROR_OUT_OF_MEMORY;
    }
    for (size_t t = 0; t < work->task_count; ++t) {
        const pact_task *task = &work->tasks[t];
        if (task->kind == PACT_TASK_ADD) {
            ++remaining[task->dependencies[0]];
            ++remaining[task->dependencies[1]];
        }
    }
    pact_result status = PACT_SUCCESS;
    for (size_t t = 0; t < work->task_count; ++t) {
        const pact_task *task = &work->tasks[t];
        pact_tensor *node = graph->nodes[task->node_index];
        if (task->kind == PACT_TASK_SOURCE) {
            values[t] = (float *)(node->host_data + work->start * sizeof(float));
            continue;
        }
        float *destination = t == graph->count - 1
            ? (float *)(work->output + work->start * sizeof(float))
            : malloc(work->count * sizeof(float));
        if (destination == NULL) { status = PACT_ERROR_OUT_OF_MEMORY; break; }
        size_t a_index = task->dependencies[0];
        size_t b_index = task->dependencies[1];
        float *a = values[a_index];
        float *b = values[b_index];
        if (a == NULL || b == NULL) {
            if (t != graph->count - 1) free(destination);
            status = PACT_ERROR_INTERNAL;
            break;
        }
        for (size_t i = 0; i < work->count; ++i)
            destination[i] = a[i] + b[i];
        values[t] = destination;
        size_t indices[2] = {a_index, b_index};
        for (unsigned input = 0; input < 2; ++input) {
            size_t index = indices[input];
            if (--remaining[index] == 0 && !graph->nodes[index]->materialized) {
                free(values[index]);
                values[index] = NULL;
            }
        }
    }
    for (size_t i = 0; i < graph->count; ++i)
        if (!graph->nodes[i]->materialized && i != graph->count - 1)
            free(values[i]);
    free(values);
    free(remaining);
    return status;
}

static void pact_cpu_execute_item(void *opaque)
{
    pact_cpu_work *work = opaque;
    if (work->linear != NULL && work->linear->base != NULL) {
        float *result = (float *)(work->output + work->start * sizeof(float));
        for (size_t i = 0; i < work->count; ++i) {
            size_t index = work->start + i;
            float value = work->linear->base[index];
            for (size_t step = 0; step < work->linear->count; ++step) {
                const pact_linear_step *operation = &work->linear->steps[step];
                float operand = operation->operand[index];
                value = operation->operand_first
                    ? operand + value : value + operand;
            }
            result[i] = value;
        }
        work->status = PACT_SUCCESS;
    } else work->status = pact_cpu_generic(work);
}

static pact_result pact_run_cpu(
    pact_runtime *runtime, const pact_graph *graph,
    const pact_linear_program *linear, size_t start, size_t count,
    unsigned char *output, double *out_us, uint32_t *out_tasks)
{
    *out_us = 0.0;
    *out_tasks = 0;
    if (count == 0) return PACT_SUCCESS;
    pact_task *tasks = NULL;
    size_t task_count = 0;
    pact_result status = pact_make_tasks(
        graph, start, count, 0, 0, &tasks, &task_count);
    if (status != PACT_SUCCESS) return status;
    size_t workers = 1;
    size_t operations = graph->count;
    if (count >= 131072 && operations > 2 &&
        count <= SIZE_MAX / operations && count * operations >= 1048576) {
        workers = runtime->devices[0].info.logical_processor_count;
        if (workers > 8) workers = 8;
        if (workers == 0) workers = 1;
    }
    size_t items = workers == 1 ? 1 : workers * 4;
    if (items > count / 16384) items = count / 16384;
    if (items == 0) items = 1;
    *out_tasks = (uint32_t)items;
    pact_cpu_work *work = calloc(items, sizeof(*work));
    if (work == NULL) { free(tasks); return PACT_ERROR_OUT_OF_MEMORY; }
    for (size_t i = 0; i < items; ++i) {
        size_t lo = start + (count / items) * i + (count % items) * i / items;
        size_t hi = start + (count / items) * (i + 1) +
                    (count % items) * (i + 1) / items;
        work[i] = (pact_cpu_work){graph, tasks, linear, task_count,
                                  lo, hi - lo, output, PACT_SUCCESS};
    }
    if (items > 1) pact_cpu_pool_prepare(runtime);
    double begin = pact_now_us();
    pact_cpu_pool_run(runtime, pact_cpu_execute_item, work, sizeof(*work), items);
    *out_us = pact_now_us() - begin;
    for (size_t i = 0; i < items; ++i)
        if (work[i].status != PACT_SUCCESS) status = work[i].status;
    free(work);
    free(tasks);
    return status;
}

static int pact_gpu_execute_item(void *opaque)
{
    pact_gpu_work *work = opaque;
    work->status = pact_vulkan_execute_tasks(
        work->runtime, work->device_index, work->graph, work->tasks,
        work->task_count, work->output, &work->timing);
    return 0;
}

static int pact_sample_matches(
    const pact_performance_sample *sample, uint32_t additions,
    size_t elements)
{
    return sample->valid && sample->additions == additions &&
        elements >= sample->sample_elements / 2 &&
        (sample->sample_elements > SIZE_MAX / 2 ||
         elements <= sample->sample_elements * 2);
}

static void pact_update_sample(
    pact_performance_sample *sample, uint32_t additions,
    size_t elements, double microseconds)
{
    if (elements == 0 || microseconds <= 0.0) return;
    double rate = microseconds / (double)elements;
    if (pact_sample_matches(sample, additions, elements)) {
        rate = sample->microseconds_per_element * 0.7 + rate * 0.3;
        ++sample->observations;
    } else sample->observations = 1;
    sample->additions = additions;
    sample->microseconds_per_element = rate;
    sample->sample_elements = elements;
    sample->valid = 1;
}

static void pact_update_cpu_sample(
    pact_performance_sample *sample, uint32_t additions,
    size_t elements, double microseconds)
{
    if (elements == 0 || microseconds <= 0.0) return;
    double rate = microseconds / (double)elements;
    if (pact_sample_matches(sample, additions, elements)) {
        if (rate < sample->microseconds_per_element)
            sample->microseconds_per_element = rate;
        ++sample->observations;
        sample->sample_elements = elements;
    } else {
        sample->additions = additions;
        sample->microseconds_per_element = rate;
        sample->sample_elements = elements;
        sample->observations = 1;
        sample->valid = 1;
    }
}

static size_t pact_choose_gpu_counts(
    pact_runtime *runtime, uint32_t additions, size_t elements,
    size_t *counts)
{
    uint32_t eligible = 0;
    int needs_pilot = 0;
    for (uint32_t i = 0; i < runtime->device_count; ++i) {
        pact_device_record *device = &runtime->devices[i];
        if (device->info.backend == PACT_BACKEND_VULKAN &&
            device->execution_status == PACT_SUCCESS) {
            ++eligible;
            if (!device->performance.valid ||
                device->performance.additions != additions)
                needs_pilot = 1;
        }
    }
    if (eligible == 0 || additions == 0 || elements < 1048576 ||
        elements < 4194304u / additions) return 0;
    if (!pact_sample_matches(&runtime->cpu_performance, additions, elements) ||
        runtime->cpu_performance.observations < 2)
        return 0;
    /* A GPU pilot has a fixed transfer and setup cost. Avoid it for short reads. */
    if (needs_pilot &&
        runtime->cpu_performance.microseconds_per_element * elements < 1000.0)
        return 0;
    if (runtime->mixed_losses >= 2 &&
        runtime->mixed_additions == additions &&
        elements >= runtime->mixed_elements / 2 &&
        (runtime->mixed_elements > SIZE_MAX / 2 ||
         elements <= runtime->mixed_elements * 2))
        return 0;
    if (needs_pilot) {
        size_t pilot = elements / (8u * eligible);
        if (pilot > 262144) pilot = 262144;
        pilot = (pilot / 256) * 256;
        if (pilot < 16384) return 0;
        size_t sum = 0;
        for (uint32_t i = 0; i < runtime->device_count; ++i)
            if (runtime->devices[i].info.backend == PACT_BACKEND_VULKAN &&
                runtime->devices[i].execution_status == PACT_SUCCESS) {
                counts[i] = pilot;
                sum += pilot;
            }
        return sum;
    }
    double cpu_rate = runtime->cpu_performance.microseconds_per_element;
    double throughput = 1.0 / cpu_rate;
    for (uint32_t i = 0; i < runtime->device_count; ++i) {
        pact_performance_sample *sample = &runtime->devices[i].performance;
        if (sample->valid && sample->additions == additions &&
            runtime->devices[i].execution_status == PACT_SUCCESS)
            throughput += 1.0 / sample->microseconds_per_element;
    }
    double predicted_cpu = elements * cpu_rate;
    double predicted_mixed = elements / throughput;
    if (predicted_mixed >= predicted_cpu * 0.90) return 0;
    size_t sum = 0;
    for (uint32_t i = 0; i < runtime->device_count; ++i) {
        pact_performance_sample *sample = &runtime->devices[i].performance;
        if (sample->valid && sample->additions == additions &&
            runtime->devices[i].execution_status == PACT_SUCCESS) {
            size_t count = (size_t)(predicted_mixed /
                                     sample->microseconds_per_element);
            count = (count / 256) * 256;
            if (count > elements - sum) count = elements - sum;
            counts[i] = count;
            sum += count;
        }
    }
    return sum;
}

pact_result pact_plan_execute(
    pact_tensor *root, const pact_graph *graph, unsigned char *output)
{
    if (graph->count == 0 || graph->nodes[graph->count - 1] != root)
        return PACT_ERROR_INTERNAL;
    pact_runtime *runtime = root->runtime;
    pact_execution_info *stats = &runtime->last_execution;
    *stats = (pact_execution_info){0};
    stats->struct_size = sizeof(*stats);
    size_t elements = root->element_count;
    if (elements == 0) return PACT_SUCCESS;
    uint32_t additions = 0;
    for (size_t i = 0; i < graph->count; ++i)
        if (!graph->nodes[i]->materialized) ++additions;
    pact_linear_program linear = {0};
    pact_result compiled = pact_linear_compile(graph, &linear);
    if (compiled == PACT_ERROR_OUT_OF_MEMORY) return compiled;
    size_t *gpu_counts = calloc(runtime->device_count, sizeof(*gpu_counts));
    pact_gpu_work *gpu = calloc(runtime->device_count, sizeof(*gpu));
    if (gpu_counts == NULL || gpu == NULL) {
        free(gpu_counts); free(gpu); free(linear.steps);
        return PACT_ERROR_OUT_OF_MEMORY;
    }
    size_t gpu_total = graph->count <= 64
        ? pact_choose_gpu_counts(runtime, additions, elements, gpu_counts) : 0;
    double plan_begin = pact_now_us();
    size_t cpu_count = elements - gpu_total;
    size_t cursor = cpu_count;
    pact_result status = PACT_SUCCESS;
    for (uint32_t i = 0; i < runtime->device_count; ++i) {
        if (gpu_counts[i] == 0) continue;
        pact_gpu_work *work = &gpu[i];
        work->runtime = runtime;
        work->graph = graph;
        work->device_index = i;
        work->start = cursor;
        work->count = gpu_counts[i];
        work->output = output;
        cursor += gpu_counts[i];
        status = pact_make_tasks(graph, work->start, work->count, i, 1,
                                 &work->tasks, &work->task_count);
        if (status != PACT_SUCCESS) break;
        if (pact_thread_start(&work->thread, pact_gpu_execute_item, work))
            work->started = 1;
        else {
            work->status = PACT_ERROR_OUT_OF_MEMORY;
            stats->vulkan_fallback = work->status;
        }
    }
    double cpu_us = 0.0;
    uint32_t cpu_tasks = 0;
    if (status == PACT_SUCCESS)
        status = pact_run_cpu(runtime, graph, &linear, 0, cpu_count,
                              output, &cpu_us, &cpu_tasks);
    stats->cpu_time_us = cpu_us;
    stats->cpu_tasks = cpu_tasks;
    if (status == PACT_SUCCESS && cpu_count != 0) {
        stats->cpu_shards = 1;
        stats->cpu_elements = cpu_count;
        if (gpu_total == 0)
            pact_update_cpu_sample(&runtime->cpu_performance,
                                   additions, cpu_count, cpu_us);
    }
    for (uint32_t i = 0; i < runtime->device_count; ++i) {
        pact_gpu_work *work = &gpu[i];
        if (work->started) pact_thread_join(work->thread);
        if (work->started && work->status == PACT_SUCCESS) {
            ++stats->vulkan_shards;
            stats->vulkan_elements += work->count;
            pact_update_sample(&runtime->devices[i].performance,
                               additions, work->count, work->timing.active_us);
#define MAX_TIMING(field) do { \
    if (work->timing.field > stats->vulkan_##field) \
        stats->vulkan_##field = work->timing.field; \
} while (0)
            MAX_TIMING(active_us);
            MAX_TIMING(setup_us);
            MAX_TIMING(upload_us);
            MAX_TIMING(record_us);
            MAX_TIMING(queue_us);
            MAX_TIMING(download_us);
#undef MAX_TIMING
        } else if (work->count != 0) {
            if (work->started) stats->vulkan_fallback = work->status;
            if (status == PACT_SUCCESS) {
                double fallback_us = 0.0;
                uint32_t fallback_tasks = 0;
                status = pact_run_cpu(runtime, graph, &linear,
                    work->start, work->count, output, &fallback_us, &fallback_tasks);
                stats->cpu_time_us += fallback_us;
                stats->cpu_tasks += fallback_tasks;
                if (status == PACT_SUCCESS) {
                    ++stats->cpu_shards;
                    stats->cpu_elements += work->count;
                }
            }
        }
        free(work->tasks);
    }
    if (gpu_total != 0 && status == PACT_SUCCESS &&
        runtime->cpu_performance.valid &&
        runtime->cpu_performance.additions == additions) {
        double mixed_us = pact_now_us() - plan_begin;
        double cpu_expected = runtime->cpu_performance.microseconds_per_element * elements;
        if (runtime->mixed_additions != additions ||
            elements < runtime->mixed_elements / 2 ||
            (runtime->mixed_elements <= SIZE_MAX / 2 &&
             elements > runtime->mixed_elements * 2)) {
            runtime->mixed_additions = additions;
            runtime->mixed_elements = elements;
            runtime->mixed_losses = 0;
        }
        if (mixed_us > cpu_expected * 1.05) ++runtime->mixed_losses;
        else runtime->mixed_losses = 0;
    }
    free(gpu_counts);
    free(gpu);
    free(linear.steps);
    return status;
}
