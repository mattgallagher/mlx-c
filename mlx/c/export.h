/* Copyright © 2023-2025 Apple Inc.                   */

#ifndef MLX_EXPORT_H
#define MLX_EXPORT_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "mlx/c/array.h"
#include "mlx/c/closure.h"
#include "mlx/c/distributed_group.h"
#include "mlx/c/io_types.h"
#include "mlx/c/map.h"
#include "mlx/c/stream.h"
#include "mlx/c/string.h"
#include "mlx/c/vector.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup export Function serialization
 */
/**@{*/
int mlx_export_function(
    const char* file,
    const mlx_closure fun,
    const mlx_vector_array args,
    bool shapeless);
int mlx_export_function_kwargs(
    const char* file,
    const mlx_closure_kwargs fun,
    const mlx_vector_array args,
    const mlx_map_string_to_array kwargs,
    bool shapeless);

typedef struct mlx_function_exporter_ {
  void* ctx;
} mlx_function_exporter;
mlx_function_exporter mlx_function_exporter_new(
    const char* file,
    const mlx_closure fun,
    bool shapeless);
int mlx_function_exporter_free(mlx_function_exporter xfunc);
int mlx_function_exporter_apply(
    const mlx_function_exporter xfunc,
    const mlx_vector_array args);
int mlx_function_exporter_apply_kwargs(
    const mlx_function_exporter xfunc,
    const mlx_vector_array args,
    const mlx_map_string_to_array kwargs);

typedef struct mlx_imported_function_ {
  void* ctx;
} mlx_imported_function;
mlx_imported_function mlx_imported_function_new(const char* file);
int mlx_imported_function_free(mlx_imported_function xfunc);
int mlx_imported_function_apply(
    mlx_vector_array* res,
    const mlx_imported_function xfunc,
    const mlx_vector_array args);
int mlx_imported_function_apply_kwargs(
    mlx_vector_array* res,
    const mlx_imported_function xfunc,
    const mlx_vector_array args,
    const mlx_map_string_to_array kwargs);

/**
 * \defgroup export_callback Callback-mode function serialization
 *
 * C projection of the callback overload of
 * mlx::core::export_function (mlx/export.h). Each invocation of the callback
 * receives one mlx_export_callback_input describing an event in the export
 * stream: "inputs", "keyword_inputs", "outputs", "constants", or one
 * "primitive" record per primitive in the captured graph.
 *
 * The C++ surface is:
 *
 *     using ExportCallbackInput = std::unordered_map<
 *         std::string,
 *         std::variant<
 *             std::vector<std::tuple<std::string, Shape, Dtype>>,
 *             std::vector<std::pair<std::string, array>>,
 *             std::vector<std::pair<std::string, std::string>>,
 *             std::vector<StateT>,
 *             std::string>>;
 *
 * Lifetime: every handle exposed inside the callback (input, value, state,
 * char*, raw int32/int64/size pointers) is borrowed and only valid for the
 * duration of that callback invocation. Copy out anything you need to keep.
 * mlx_array handles obtained via mlx_export_callback_value_named_arrays_get
 * are new references owned by the caller and must be freed with
 * mlx_array_free.
 */
/**@{*/

/** Discriminator for the outer ExportCallbackInput variant. */
typedef enum mlx_export_callback_value_kind_ {
  MLX_EXPORT_VALUE_TENSOR_SPECS = 0,     /**< vector<tuple<string,Shape,Dtype>> */
  MLX_EXPORT_VALUE_NAMED_ARRAYS = 1,     /**< vector<pair<string,array>>        */
  MLX_EXPORT_VALUE_NAMED_STRINGS = 2,    /**< vector<pair<string,string>>       */
  MLX_EXPORT_VALUE_PRIMITIVE_STATES = 3, /**< vector<StateT>                    */
  MLX_EXPORT_VALUE_STRING = 4,           /**< string                            */
} mlx_export_callback_value_kind;

/** Discriminator for the inner StateT variant (per mlx/export.h:18-32). */
typedef enum mlx_export_state_kind_ {
  MLX_EXPORT_STATE_BOOL = 0,
  MLX_EXPORT_STATE_INT = 1,
  MLX_EXPORT_STATE_SIZE = 2,
  MLX_EXPORT_STATE_FLOAT = 3,
  MLX_EXPORT_STATE_DOUBLE = 4,
  MLX_EXPORT_STATE_DTYPE = 5,
  MLX_EXPORT_STATE_SHAPE = 6,              /**< vector<int32>            */
  MLX_EXPORT_STATE_STRIDES = 7,            /**< vector<int64>            */
  MLX_EXPORT_STATE_INT_ARRAY = 8,          /**< vector<int>              */
  MLX_EXPORT_STATE_SIZE_ARRAY = 9,         /**< vector<size_t>           */
  MLX_EXPORT_STATE_TRIPLE_BOOL_ARRAY = 10, /**< vector<tuple<bool,bool,bool>> */
  MLX_EXPORT_STATE_MIXED_SCALAR_ARRAY = 11, /**< vector<variant<bool,int,float>> */
  MLX_EXPORT_STATE_OPTIONAL_FLOAT = 12,
  MLX_EXPORT_STATE_STRING = 13,
} mlx_export_state_kind;

/** Discriminator for the inner variant<bool,int,float>. */
typedef enum mlx_export_mixed_scalar_kind_ {
  MLX_EXPORT_MIXED_BOOL = 0,
  MLX_EXPORT_MIXED_INT = 1,
  MLX_EXPORT_MIXED_FLOAT = 2,
} mlx_export_mixed_scalar_kind;

/** Opaque, callback-scoped handle to ExportCallbackInput. */
typedef struct mlx_export_callback_input_ {
  void* ctx;
} mlx_export_callback_input;

/** Opaque, callback-scoped handle to one entry value in ExportCallbackInput. */
typedef struct mlx_export_callback_value_ {
  void* ctx;
} mlx_export_callback_value;

/** Opaque, callback-scoped handle to one StateT value. */
typedef struct mlx_export_state_ {
  void* ctx;
} mlx_export_state;

/**
 * Number of entries in the input map.
 *
 * The Python ExportCallback payload uses the keys "type", "inputs",
 * "keyword_inputs", "outputs", "constants", "name", "arguments". Iteration
 * order matches std::unordered_map iteration order (unspecified); use keys to
 * dispatch.
 */
size_t mlx_export_callback_input_size(const mlx_export_callback_input input);

/**
 * Borrow the key at index. Valid for the callback's lifetime only.
 */
const char* mlx_export_callback_input_key(
    const mlx_export_callback_input input,
    size_t index);

/**
 * Borrow the value at index. Valid for the callback's lifetime only.
 */
mlx_export_callback_value mlx_export_callback_input_value(
    const mlx_export_callback_input input,
    size_t index);

/** Variant tag of the value. */
mlx_export_callback_value_kind mlx_export_callback_value_get_kind(
    const mlx_export_callback_value value);

/* --- MLX_EXPORT_VALUE_TENSOR_SPECS --- */
size_t mlx_export_callback_value_tensor_specs_size(
    const mlx_export_callback_value value);
const char* mlx_export_callback_value_tensor_specs_name(
    const mlx_export_callback_value value,
    size_t index);
/** Returns 0 on success; writes a borrowed pointer/length pair. */
int mlx_export_callback_value_tensor_specs_shape(
    const int32_t** out_data,
    size_t* out_size,
    const mlx_export_callback_value value,
    size_t index);
int mlx_export_callback_value_tensor_specs_dtype(
    mlx_dtype* out,
    const mlx_export_callback_value value,
    size_t index);

/* --- MLX_EXPORT_VALUE_NAMED_ARRAYS --- */
size_t mlx_export_callback_value_named_arrays_size(
    const mlx_export_callback_value value);
const char* mlx_export_callback_value_named_arrays_name(
    const mlx_export_callback_value value,
    size_t index);
/**
 * Writes a new owned mlx_array reference (caller must mlx_array_free).
 * Returns 0 on success.
 */
int mlx_export_callback_value_named_arrays_get(
    mlx_array* out,
    const mlx_export_callback_value value,
    size_t index);

/* --- MLX_EXPORT_VALUE_NAMED_STRINGS --- */
size_t mlx_export_callback_value_named_strings_size(
    const mlx_export_callback_value value);
const char* mlx_export_callback_value_named_strings_first(
    const mlx_export_callback_value value,
    size_t index);
const char* mlx_export_callback_value_named_strings_second(
    const mlx_export_callback_value value,
    size_t index);

/* --- MLX_EXPORT_VALUE_PRIMITIVE_STATES --- */
size_t mlx_export_callback_value_states_size(
    const mlx_export_callback_value value);
mlx_export_state mlx_export_callback_value_states_at(
    const mlx_export_callback_value value,
    size_t index);

/* --- MLX_EXPORT_VALUE_STRING --- */
const char* mlx_export_callback_value_string(
    const mlx_export_callback_value value);

/* --- StateT accessors --- */
mlx_export_state_kind mlx_export_state_get_kind(const mlx_export_state state);

int mlx_export_state_bool(bool* out, const mlx_export_state state);
int mlx_export_state_int(int* out, const mlx_export_state state);
int mlx_export_state_size(size_t* out, const mlx_export_state state);
int mlx_export_state_float(float* out, const mlx_export_state state);
int mlx_export_state_double(double* out, const mlx_export_state state);
int mlx_export_state_dtype(mlx_dtype* out, const mlx_export_state state);

int mlx_export_state_shape(
    const int32_t** out_data,
    size_t* out_size,
    const mlx_export_state state);
int mlx_export_state_strides(
    const int64_t** out_data,
    size_t* out_size,
    const mlx_export_state state);
int mlx_export_state_int_array(
    const int** out_data,
    size_t* out_size,
    const mlx_export_state state);
int mlx_export_state_size_array(
    const size_t** out_data,
    size_t* out_size,
    const mlx_export_state state);

int mlx_export_state_triple_bool_array_size(
    size_t* out_size,
    const mlx_export_state state);
int mlx_export_state_triple_bool_array_at(
    bool* out0,
    bool* out1,
    bool* out2,
    const mlx_export_state state,
    size_t index);

int mlx_export_state_mixed_scalar_array_size(
    size_t* out_size,
    const mlx_export_state state);
int mlx_export_state_mixed_scalar_array_kind(
    mlx_export_mixed_scalar_kind* out,
    const mlx_export_state state,
    size_t index);
int mlx_export_state_mixed_scalar_array_bool(
    bool* out,
    const mlx_export_state state,
    size_t index);
int mlx_export_state_mixed_scalar_array_int(
    int* out,
    const mlx_export_state state,
    size_t index);
int mlx_export_state_mixed_scalar_array_float(
    float* out,
    const mlx_export_state state,
    size_t index);

/**
 * Returns true if the optional is engaged. mlx_export_state_optional_float
 * writes the value only when this is true.
 */
bool mlx_export_state_optional_float_has_value(const mlx_export_state state);
int mlx_export_state_optional_float(float* out, const mlx_export_state state);

const char* mlx_export_state_string(const mlx_export_state state);

/* --- Callback closure type --- */

/**
 * User callback signature. The input handle is borrowed; do not retain it
 * past the call. The payload is the same pointer passed to
 * mlx_closure_export_callback_new.
 */
typedef void (*mlx_export_callback_fun)(
    const mlx_export_callback_input input,
    void* payload);

typedef struct mlx_closure_export_callback_ {
  void* ctx;
} mlx_closure_export_callback;

/**
 * Create a closure from a C function + payload + optional destructor.
 *
 * @param fun     callback invoked once per event in the export stream
 * @param payload opaque pointer passed back to fun on each invocation
 * @param dtor    optional destructor invoked when the closure is released;
 *                if NULL, the payload is not owned
 */
mlx_closure_export_callback mlx_closure_export_callback_new(
    mlx_export_callback_fun fun,
    void* payload,
    void (*dtor)(void*));
int mlx_closure_export_callback_free(mlx_closure_export_callback cls);

/**
 * Run `fun` with `args` and `kwargs`, streaming primitive events to
 * `callback`. Mirrors mlx::core::export_function(callback, ...) declared
 * at mlx/export.h:128-133.
 *
 * @return 0 on success, non-zero on error (mlx_error captures details)
 */
int mlx_export_function_callback(
    const mlx_closure_export_callback callback,
    const mlx_closure_kwargs fun,
    const mlx_vector_array args,
    const mlx_map_string_to_array kwargs,
    bool shapeless);

/**@}*/
/**@}*/

#ifdef __cplusplus
}
#endif

#endif
