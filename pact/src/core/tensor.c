#include "tensor_internal.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

static pact_result pact_tensor_new(
    pact_runtime *runtime, pact_dtype dtype, const uint64_t *shape, uint32_t rank,
    pact_tensor **out_tensor)
{
    if (runtime->next_value_id == UINT64_MAX) {
        return PACT_ERROR_INTERNAL;
    }
    pact_tensor *tensor = (pact_tensor *)calloc(1, sizeof(*tensor));
    if (tensor == NULL) {
        return PACT_ERROR_OUT_OF_MEMORY;
    }
    if (rank != 0) {
        size_t shape_bytes = (size_t)rank * sizeof(uint64_t);
        if (shape_bytes / sizeof(uint64_t) != rank) {
            free(tensor);
            return PACT_ERROR_OUT_OF_MEMORY;
        }
        tensor->shape = (uint64_t *)malloc(shape_bytes);
        if (tensor->shape == NULL) {
            free(tensor);
            return PACT_ERROR_OUT_OF_MEMORY;
        }
        memcpy(tensor->shape, shape, shape_bytes);
    }
    tensor->runtime = runtime;
    pact_runtime_retain_internal(runtime);
    tensor->ref_count = 1;
    tensor->value_id = runtime->next_value_id++;
    tensor->version = 1;
    tensor->dtype = dtype;
    tensor->rank = rank;
    tensor->element_count = 1;
    for (uint32_t i = 0; i < rank; ++i) {
        uint64_t dim = shape[i];
        if (dim > SIZE_MAX ||
            (dim != 0 && tensor->element_count > SIZE_MAX / (size_t)dim)) {
            pact_tensor_release(tensor);
            return PACT_ERROR_OUT_OF_MEMORY;
        }
        tensor->element_count *= (size_t)dim;
    }
    if (tensor->element_count > SIZE_MAX / sizeof(float)) {
        pact_tensor_release(tensor);
        return PACT_ERROR_OUT_OF_MEMORY;
    }
    tensor->byte_count = tensor->element_count * sizeof(float);
    tensor->failure = PACT_SUCCESS;
    *out_tensor = tensor;
    return PACT_SUCCESS;
}

pact_result pact_tensor_from_data(
    pact_runtime *runtime, pact_dtype dtype, const uint64_t *shape, uint32_t rank,
    const void *data, size_t data_bytes, pact_tensor **out_tensor)
{
    if (out_tensor == NULL) {
        return PACT_ERROR_INVALID_ARGUMENT;
    }
    *out_tensor = NULL;
    if (runtime == NULL || (rank != 0 && shape == NULL)) {
        return PACT_ERROR_INVALID_ARGUMENT;
    }
    if (dtype != PACT_DTYPE_F32) {
        return PACT_ERROR_UNSUPPORTED_DTYPE;
    }

    pact_tensor *tensor = NULL;
    pact_result result = pact_tensor_new(runtime, dtype, shape, rank, &tensor);
    if (result != PACT_SUCCESS) {
        return result;
    }
    if (data_bytes != tensor->byte_count) {
        pact_tensor_release(tensor);
        return PACT_ERROR_BUFFER_SIZE;
    }
    if (data_bytes != 0 && data == NULL) {
        pact_tensor_release(tensor);
        return PACT_ERROR_INVALID_ARGUMENT;
    }
    if (data_bytes != 0) {
        tensor->host_data = (unsigned char *)malloc(data_bytes);
        if (tensor->host_data == NULL) {
            pact_tensor_release(tensor);
            return PACT_ERROR_OUT_OF_MEMORY;
        }
        memcpy(tensor->host_data, data, data_bytes);
    }
    tensor->op = PACT_OP_INPUT;
    tensor->materialized = 1;
    *out_tensor = tensor;
    return PACT_SUCCESS;
}

pact_result pact_tensor_add(pact_tensor *a, pact_tensor *b, pact_tensor **out_tensor)
{
    if (out_tensor == NULL) {
        return PACT_ERROR_INVALID_ARGUMENT;
    }
    *out_tensor = NULL;
    if (a == NULL || b == NULL || a->runtime != b->runtime) {
        return PACT_ERROR_INVALID_ARGUMENT;
    }
    if (a->failure != PACT_SUCCESS) return a->failure;
    if (b->failure != PACT_SUCCESS) return b->failure;
    if (a->dtype != PACT_DTYPE_F32 || b->dtype != PACT_DTYPE_F32) {
        return PACT_ERROR_UNSUPPORTED_DTYPE;
    }
    if (a->rank != b->rank ||
        (a->rank != 0 && memcmp(a->shape, b->shape,
                                (size_t)a->rank * sizeof(uint64_t)) != 0)) {
        return PACT_ERROR_SHAPE_MISMATCH;
    }
    pact_tensor *tensor = NULL;
    pact_result result = pact_tensor_new(a->runtime, a->dtype, a->shape, a->rank, &tensor);
    if (result != PACT_SUCCESS) {
        return result;
    }
    tensor->op = PACT_OP_ADD;
    tensor->input_a = a;
    tensor->input_b = b;
    pact_tensor_retain(a);
    pact_tensor_retain(b);
    *out_tensor = tensor;
    return PACT_SUCCESS;
}

pact_result pact_tensor_read(pact_tensor *tensor, void *out_data, size_t out_bytes)
{
    if (tensor == NULL || (out_bytes != 0 && out_data == NULL)) {
        return PACT_ERROR_INVALID_ARGUMENT;
    }
    if (out_bytes != tensor->byte_count) {
        return PACT_ERROR_BUFFER_SIZE;
    }
    if (tensor->failure != PACT_SUCCESS) {
        return tensor->failure;
    }
    if (!tensor->materialized) {
        unsigned char *output = NULL;
        if (tensor->byte_count != 0) {
            output = (unsigned char *)malloc(tensor->byte_count);
            if (output == NULL) {
                return PACT_ERROR_OUT_OF_MEMORY;
            }
        }
        pact_graph graph = {0};
        pact_result result = pact_graph_collect(tensor, &graph);
        if (result == PACT_SUCCESS) {
            result = pact_plan_execute(tensor, &graph, output);
        }
        pact_graph_release(&graph);
        if (result != PACT_SUCCESS) {
            free(output);
            tensor->failure = result;
            return result;
        }
        tensor->host_data = output;
        tensor->materialized = 1;
        pact_tensor *a = tensor->input_a;
        pact_tensor *b = tensor->input_b;
        tensor->input_a = NULL;
        tensor->input_b = NULL;
        pact_tensor_release(a);
        pact_tensor_release(b);
    }
    if (out_bytes != 0) {
        memcpy(out_data, tensor->host_data, out_bytes);
    }
    return PACT_SUCCESS;
}

pact_result pact_tensor_get_rank(const pact_tensor *tensor, uint32_t *out_rank)
{
    if (tensor == NULL || out_rank == NULL) {
        return PACT_ERROR_INVALID_ARGUMENT;
    }
    *out_rank = tensor->rank;
    return PACT_SUCCESS;
}

pact_result pact_tensor_get_shape(
    const pact_tensor *tensor, uint64_t *out_shape, uint32_t shape_capacity)
{
    if (tensor == NULL || shape_capacity < tensor->rank ||
        (tensor->rank != 0 && out_shape == NULL)) {
        return PACT_ERROR_INVALID_ARGUMENT;
    }
    if (tensor->rank != 0) {
        memcpy(out_shape, tensor->shape, (size_t)tensor->rank * sizeof(uint64_t));
    }
    return PACT_SUCCESS;
}

pact_result pact_tensor_get_size_bytes(const pact_tensor *tensor, size_t *out_bytes)
{
    if (tensor == NULL || out_bytes == NULL) {
        return PACT_ERROR_INVALID_ARGUMENT;
    }
    *out_bytes = tensor->byte_count;
    return PACT_SUCCESS;
}

void pact_tensor_retain(pact_tensor *tensor)
{
    if (tensor != NULL && tensor->ref_count < SIZE_MAX) {
        ++tensor->ref_count;
    }
}

void pact_tensor_release(pact_tensor *tensor)
{
    if (tensor == NULL || --tensor->ref_count != 0) {
        return;
    }
    pact_tensor *pending = tensor;
    pending->release_next = NULL;
    while (pending != NULL) {
        pact_tensor *node = pending;
        pending = node->release_next;
        pact_tensor *inputs[2] = {node->input_a, node->input_b};
        for (unsigned int i = 0; i < 2; ++i) {
            pact_tensor *input = inputs[i];
            if (input != NULL && --input->ref_count == 0) {
                input->release_next = pending;
                pending = input;
            }
        }
        pact_runtime *runtime = node->runtime;
        free(node->shape);
        free(node->host_data);
        free(node);
        pact_runtime_release_internal(runtime);
    }
}
