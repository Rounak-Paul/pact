#include "tensor_internal.h"

#include <stdlib.h>

typedef struct pact_graph_frame {
    pact_tensor *node;
    unsigned char next_input;
} pact_graph_frame;

static pact_result pact_graph_push(
    pact_graph_frame **frames, size_t *count, size_t *capacity, pact_tensor *node)
{
    if (*count == *capacity) {
        size_t next = *capacity == 0 ? 16 : *capacity * 2;
        if (next < *capacity || next > SIZE_MAX / sizeof(**frames)) {
            return PACT_ERROR_OUT_OF_MEMORY;
        }
        pact_graph_frame *grown = (pact_graph_frame *)realloc(
            *frames, next * sizeof(**frames));
        if (grown == NULL) {
            return PACT_ERROR_OUT_OF_MEMORY;
        }
        *frames = grown;
        *capacity = next;
    }
    (*frames)[(*count)++] = (pact_graph_frame){node, 0};
    return PACT_SUCCESS;
}

static pact_result pact_graph_append(
    pact_graph *graph, size_t *capacity, pact_tensor *node)
{
    if (graph->count == *capacity) {
        size_t next = *capacity == 0 ? 16 : *capacity * 2;
        if (next < *capacity || next > SIZE_MAX / sizeof(*graph->nodes)) {
            return PACT_ERROR_OUT_OF_MEMORY;
        }
        pact_tensor **grown = (pact_tensor **)realloc(
            graph->nodes, next * sizeof(*graph->nodes));
        if (grown == NULL) {
            return PACT_ERROR_OUT_OF_MEMORY;
        }
        graph->nodes = grown;
        *capacity = next;
    }
    node->plan_index = graph->count;
    graph->nodes[graph->count++] = node;
    return PACT_SUCCESS;
}

pact_result pact_graph_collect(pact_tensor *root, pact_graph *out_graph)
{
    if (root == NULL || out_graph == NULL) {
        return PACT_ERROR_INVALID_ARGUMENT;
    }
    *out_graph = (pact_graph){0};
    uint64_t epoch = ++root->runtime->visit_epoch;
    if (epoch == 0) {
        return PACT_ERROR_INTERNAL;
    }

    pact_graph_frame *frames = NULL;
    size_t frame_count = 0;
    size_t frame_capacity = 0;
    size_t graph_capacity = 0;
    pact_result result = pact_graph_push(
        &frames, &frame_count, &frame_capacity, root);
    if (result != PACT_SUCCESS) {
        return result;
    }
    root->visit_epoch = epoch;
    root->visit_state = 1;

    while (frame_count != 0) {
        pact_graph_frame *frame = &frames[frame_count - 1];
        pact_tensor *node = frame->node;
        if (node->failure != PACT_SUCCESS) {
            result = node->failure;
            break;
        }
        if (!node->materialized && node->op == PACT_OP_ADD && frame->next_input < 2) {
            pact_tensor *input = frame->next_input++ == 0 ? node->input_a : node->input_b;
            if (input == NULL) {
                result = PACT_ERROR_INTERNAL;
                break;
            }
            if (input->visit_epoch == epoch) {
                if (input->visit_state == 1) {
                    result = PACT_ERROR_INTERNAL;
                    break;
                }
                continue;
            }
            input->visit_epoch = epoch;
            input->visit_state = 1;
            result = pact_graph_push(
                &frames, &frame_count, &frame_capacity, input);
            if (result != PACT_SUCCESS) {
                break;
            }
            continue;
        }
        node->visit_state = 2;
        result = pact_graph_append(out_graph, &graph_capacity, node);
        if (result != PACT_SUCCESS) {
            break;
        }
        --frame_count;
    }

    free(frames);
    if (result != PACT_SUCCESS) {
        pact_graph_release(out_graph);
    }
    return result;
}

void pact_graph_release(pact_graph *graph)
{
    if (graph == NULL) {
        return;
    }
    free(graph->nodes);
    *graph = (pact_graph){0};
}
