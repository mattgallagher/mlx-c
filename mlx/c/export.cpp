#include "mlx/c/export.h"
#include "mlx/c/error.h"
#include "mlx/c/private/array.h"
#include "mlx/c/private/enums.h"
#include "mlx/c/private/mlx.h"
#include "mlx/export.h"

#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

extern "C" int mlx_export_function(
    const char* file,
    const mlx_closure fun,
    const mlx_vector_array args,
    bool shapeless) {
  try {
    mlx::core::export_function(
        std::string(file),
        mlx_closure_get_(fun),
        mlx_vector_array_get_(args),
        shapeless);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_export_function_kwargs(
    const char* file,
    const mlx_closure_kwargs fun,
    const mlx_vector_array args,
    const mlx_map_string_to_array kwargs,
    bool shapeless) {
  try {
    mlx::core::export_function(
        std::string(file),
        mlx_closure_kwargs_get_(fun),
        mlx_vector_array_get_(args),
        mlx_map_string_to_array_get_(kwargs),
        shapeless);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" mlx_function_exporter mlx_function_exporter_new(
    const char* file,
    const mlx_closure fun,
    bool shapeless) {
  try {
    return mlx_function_exporter_new_(
        mlx::core::exporter(
            std::string(file), mlx_closure_get_(fun), shapeless));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return {nullptr};
  }
}
extern "C" int mlx_function_exporter_free(mlx_function_exporter xfunc) {
  try {
    mlx_function_exporter_free_(xfunc);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_function_exporter_apply(
    const mlx_function_exporter xfunc,
    const mlx_vector_array args) {
  try {
    mlx_function_exporter_get_(xfunc)(mlx_vector_array_get_(args));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_function_exporter_apply_kwargs(
    const mlx_function_exporter xfunc,
    const mlx_vector_array args,
    const mlx_map_string_to_array kwargs) {
  try {
    mlx_function_exporter_get_(xfunc)(
        mlx_vector_array_get_(args), mlx_map_string_to_array_get_(kwargs));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" mlx_imported_function mlx_imported_function_new(const char* file) {
  try {
    return mlx_imported_function_new_(
        mlx::core::import_function(std::string(file)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return {nullptr};
  }
}
extern "C" int mlx_imported_function_free(mlx_imported_function xfunc) {
  try {
    mlx_imported_function_free_(xfunc);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_imported_function_apply(
    mlx_vector_array* res,
    const mlx_imported_function xfunc,
    const mlx_vector_array args) {
  try {
    mlx_vector_array_set_(
        *res, mlx_imported_function_get_(xfunc)(mlx_vector_array_get_(args)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_imported_function_apply_kwargs(
    mlx_vector_array* res,
    const mlx_imported_function xfunc,
    const mlx_vector_array args,
    const mlx_map_string_to_array kwargs) {
  try {
    mlx_vector_array_set_(
        *res,
        mlx_imported_function_get_(xfunc)(
            mlx_vector_array_get_(args), mlx_map_string_to_array_get_(kwargs)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

// ============================================================================
// Callback-mode export — C projection of
// mlx::core::export_function(callback, ...) at mlx/export.h:116-133.
// ============================================================================

namespace {

using CppCallbackInput = mlx::core::ExportCallbackInput;
using CppCallbackEntry = CppCallbackInput::value_type;     // pair<const string, variant>
using CppCallbackValue = CppCallbackInput::mapped_type;    // variant<...>
using CppState = mlx::core::StateT;
using CppShape = mlx::core::Shape;
using CppStrides = mlx::core::Strides;
using CppTensorSpec = std::tuple<std::string, CppShape, mlx::core::Dtype>;
using CppNamedArray = std::pair<std::string, mlx::core::array>;
using CppNamedString = std::pair<std::string, std::string>;
using CppMixedScalar = std::variant<bool, int, float>;
using CppTripleBool = std::tuple<bool, bool, bool>;

// Snapshot of ExportCallbackInput entries in a stable, index-addressable
// order. Lives on the stack inside the callback shim — index access is O(1)
// and iteration order is consistent across calls within one callback.
struct InputSnapshot {
  std::vector<const CppCallbackEntry*> entries;
};

// --- Closure handle helpers (inline, file-local) ---

inline mlx_closure_export_callback mlx_closure_export_callback_new_() {
  return mlx_closure_export_callback({nullptr});
}

inline mlx_closure_export_callback mlx_closure_export_callback_new_(
    mlx::core::ExportCallback&& s) {
  return mlx_closure_export_callback(
      {new mlx::core::ExportCallback(std::move(s))});
}

inline mlx::core::ExportCallback& mlx_closure_export_callback_get_(
    mlx_closure_export_callback d) {
  if (!d.ctx) {
    throw std::runtime_error(
        "expected a non-empty mlx_closure_export_callback");
  }
  return *static_cast<mlx::core::ExportCallback*>(d.ctx);
}

inline void mlx_closure_export_callback_free_(
    mlx_closure_export_callback d) {
  if (d.ctx) {
    delete static_cast<mlx::core::ExportCallback*>(d.ctx);
  }
}

// --- Variant index helpers ---

inline const CppCallbackValue& deref_value(
    mlx_export_callback_value value) {
  return *static_cast<const CppCallbackValue*>(value.ctx);
}

inline const CppState& deref_state(mlx_export_state state) {
  return *static_cast<const CppState*>(state.ctx);
}

// `std::variant::index()` matches our public enum ordering by construction:
// see export.h MLX_EXPORT_VALUE_* / MLX_EXPORT_STATE_* / MLX_EXPORT_MIXED_*.
inline mlx_export_callback_value_kind value_kind_of(
    const CppCallbackValue& v) {
  return static_cast<mlx_export_callback_value_kind>(v.index());
}

inline mlx_export_state_kind state_kind_of(const CppState& s) {
  return static_cast<mlx_export_state_kind>(s.index());
}

inline mlx_export_mixed_scalar_kind mixed_kind_of(const CppMixedScalar& m) {
  return static_cast<mlx_export_mixed_scalar_kind>(m.index());
}

} // namespace

// --- Closure constructor / destructor ---

extern "C" mlx_closure_export_callback mlx_closure_export_callback_new(
    mlx_export_callback_fun fun,
    void* payload,
    void (*dtor)(void*)) {
  try {
    std::shared_ptr<void> cpp_payload = nullptr;
    if (dtor) {
      cpp_payload = std::shared_ptr<void>(payload, dtor);
    } else {
      cpp_payload = std::shared_ptr<void>(payload, [](void*) {});
    }
    auto cpp_callback =
        [fun, cpp_payload](const CppCallbackInput& input) {
          InputSnapshot snap;
          snap.entries.reserve(input.size());
          for (const auto& entry : input) {
            snap.entries.push_back(&entry);
          }
          mlx_export_callback_input handle{static_cast<void*>(&snap)};
          fun(handle, cpp_payload.get());
        };
    return mlx_closure_export_callback_new_(
        mlx::core::ExportCallback(std::move(cpp_callback)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return mlx_closure_export_callback_new_();
  }
}

extern "C" int mlx_closure_export_callback_free(
    mlx_closure_export_callback cls) {
  try {
    mlx_closure_export_callback_free_(cls);
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

// --- ExportCallbackInput accessors ---

extern "C" size_t mlx_export_callback_input_size(
    const mlx_export_callback_input input) {
  return static_cast<const InputSnapshot*>(input.ctx)->entries.size();
}

extern "C" const char* mlx_export_callback_input_key(
    const mlx_export_callback_input input,
    size_t index) {
  const auto& entries =
      static_cast<const InputSnapshot*>(input.ctx)->entries;
  return entries[index]->first.c_str();
}

extern "C" mlx_export_callback_value mlx_export_callback_input_value(
    const mlx_export_callback_input input,
    size_t index) {
  const auto& entries =
      static_cast<const InputSnapshot*>(input.ctx)->entries;
  const CppCallbackValue& v = entries[index]->second;
  return mlx_export_callback_value{
      const_cast<void*>(static_cast<const void*>(&v))};
}

extern "C" mlx_export_callback_value_kind mlx_export_callback_value_get_kind(
    const mlx_export_callback_value value) {
  return value_kind_of(deref_value(value));
}

// --- MLX_EXPORT_VALUE_TENSOR_SPECS ---

extern "C" size_t mlx_export_callback_value_tensor_specs_size(
    const mlx_export_callback_value value) {
  return std::get<std::vector<CppTensorSpec>>(deref_value(value)).size();
}

extern "C" const char* mlx_export_callback_value_tensor_specs_name(
    const mlx_export_callback_value value,
    size_t index) {
  return std::get<0>(
             std::get<std::vector<CppTensorSpec>>(deref_value(value))[index])
      .c_str();
}

extern "C" int mlx_export_callback_value_tensor_specs_shape(
    const int32_t** out_data,
    size_t* out_size,
    const mlx_export_callback_value value,
    size_t index) {
  try {
    const auto& spec =
        std::get<std::vector<CppTensorSpec>>(deref_value(value))[index];
    const CppShape& shape = std::get<1>(spec);
    *out_data = shape.data();
    *out_size = shape.size();
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" int mlx_export_callback_value_tensor_specs_dtype(
    mlx_dtype* out,
    const mlx_export_callback_value value,
    size_t index) {
  try {
    const auto& spec =
        std::get<std::vector<CppTensorSpec>>(deref_value(value))[index];
    *out = mlx_dtype_to_c(std::get<2>(spec));
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

// --- MLX_EXPORT_VALUE_NAMED_ARRAYS ---

extern "C" size_t mlx_export_callback_value_named_arrays_size(
    const mlx_export_callback_value value) {
  return std::get<std::vector<CppNamedArray>>(deref_value(value)).size();
}

extern "C" const char* mlx_export_callback_value_named_arrays_name(
    const mlx_export_callback_value value,
    size_t index) {
  return std::get<std::vector<CppNamedArray>>(deref_value(value))[index]
      .first.c_str();
}

extern "C" int mlx_export_callback_value_named_arrays_get(
    mlx_array* out,
    const mlx_export_callback_value value,
    size_t index) {
  try {
    const auto& named =
        std::get<std::vector<CppNamedArray>>(deref_value(value))[index];
    mlx_array_set_(*out, named.second);
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

// --- MLX_EXPORT_VALUE_NAMED_STRINGS ---

extern "C" size_t mlx_export_callback_value_named_strings_size(
    const mlx_export_callback_value value) {
  return std::get<std::vector<CppNamedString>>(deref_value(value)).size();
}

extern "C" const char* mlx_export_callback_value_named_strings_first(
    const mlx_export_callback_value value,
    size_t index) {
  return std::get<std::vector<CppNamedString>>(deref_value(value))[index]
      .first.c_str();
}

extern "C" const char* mlx_export_callback_value_named_strings_second(
    const mlx_export_callback_value value,
    size_t index) {
  return std::get<std::vector<CppNamedString>>(deref_value(value))[index]
      .second.c_str();
}

// --- MLX_EXPORT_VALUE_PRIMITIVE_STATES ---

extern "C" size_t mlx_export_callback_value_states_size(
    const mlx_export_callback_value value) {
  return std::get<std::vector<CppState>>(deref_value(value)).size();
}

extern "C" mlx_export_state mlx_export_callback_value_states_at(
    const mlx_export_callback_value value,
    size_t index) {
  const CppState& state =
      std::get<std::vector<CppState>>(deref_value(value))[index];
  return mlx_export_state{
      const_cast<void*>(static_cast<const void*>(&state))};
}

// --- MLX_EXPORT_VALUE_STRING ---

extern "C" const char* mlx_export_callback_value_string(
    const mlx_export_callback_value value) {
  return std::get<std::string>(deref_value(value)).c_str();
}

// --- StateT accessors ---

extern "C" mlx_export_state_kind mlx_export_state_get_kind(
    const mlx_export_state state) {
  return state_kind_of(deref_state(state));
}

#define MLX_STATE_SCALAR_GETTER(SUFFIX, CTYPE, CPPTYPE)                   \
  extern "C" int mlx_export_state_##SUFFIX(                                \
      CTYPE* out, const mlx_export_state state) {                          \
    try {                                                                   \
      *out = std::get<CPPTYPE>(deref_state(state));                         \
      return 0;                                                             \
    } catch (std::exception& e) {                                          \
      mlx_error(e.what());                                                  \
      return 1;                                                             \
    }                                                                       \
  }

MLX_STATE_SCALAR_GETTER(bool, bool, bool)
MLX_STATE_SCALAR_GETTER(int, int, int)
MLX_STATE_SCALAR_GETTER(size, size_t, size_t)
MLX_STATE_SCALAR_GETTER(float, float, float)
MLX_STATE_SCALAR_GETTER(double, double, double)

#undef MLX_STATE_SCALAR_GETTER

extern "C" int mlx_export_state_dtype(
    mlx_dtype* out,
    const mlx_export_state state) {
  try {
    *out = mlx_dtype_to_c(std::get<mlx::core::Dtype>(deref_state(state)));
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

#define MLX_STATE_VECTOR_GETTER(SUFFIX, CTYPE, CPPTYPE)                   \
  extern "C" int mlx_export_state_##SUFFIX(                                \
      const CTYPE** out_data,                                              \
      size_t* out_size,                                                    \
      const mlx_export_state state) {                                      \
    try {                                                                   \
      const CPPTYPE& v = std::get<CPPTYPE>(deref_state(state));            \
      *out_data = v.data();                                                 \
      *out_size = v.size();                                                 \
      return 0;                                                             \
    } catch (std::exception& e) {                                          \
      mlx_error(e.what());                                                  \
      return 1;                                                             \
    }                                                                       \
  }

MLX_STATE_VECTOR_GETTER(shape, int32_t, CppShape)
MLX_STATE_VECTOR_GETTER(strides, int64_t, CppStrides)
MLX_STATE_VECTOR_GETTER(int_array, int, std::vector<int>)
MLX_STATE_VECTOR_GETTER(size_array, size_t, std::vector<size_t>)

#undef MLX_STATE_VECTOR_GETTER

extern "C" int mlx_export_state_triple_bool_array_size(
    size_t* out_size,
    const mlx_export_state state) {
  try {
    *out_size = std::get<std::vector<CppTripleBool>>(deref_state(state)).size();
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" int mlx_export_state_triple_bool_array_at(
    bool* out0,
    bool* out1,
    bool* out2,
    const mlx_export_state state,
    size_t index) {
  try {
    const auto& tup =
        std::get<std::vector<CppTripleBool>>(deref_state(state))[index];
    *out0 = std::get<0>(tup);
    *out1 = std::get<1>(tup);
    *out2 = std::get<2>(tup);
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" int mlx_export_state_mixed_scalar_array_size(
    size_t* out_size,
    const mlx_export_state state) {
  try {
    *out_size =
        std::get<std::vector<CppMixedScalar>>(deref_state(state)).size();
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" int mlx_export_state_mixed_scalar_array_kind(
    mlx_export_mixed_scalar_kind* out,
    const mlx_export_state state,
    size_t index) {
  try {
    *out = mixed_kind_of(
        std::get<std::vector<CppMixedScalar>>(deref_state(state))[index]);
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

#define MLX_MIXED_SCALAR_GETTER(SUFFIX, CTYPE, CPPTYPE)                   \
  extern "C" int mlx_export_state_mixed_scalar_array_##SUFFIX(             \
      CTYPE* out, const mlx_export_state state, size_t index) {            \
    try {                                                                   \
      *out = std::get<CPPTYPE>(                                             \
          std::get<std::vector<CppMixedScalar>>(deref_state(state))[index]); \
      return 0;                                                             \
    } catch (std::exception& e) {                                          \
      mlx_error(e.what());                                                  \
      return 1;                                                             \
    }                                                                       \
  }

MLX_MIXED_SCALAR_GETTER(bool, bool, bool)
MLX_MIXED_SCALAR_GETTER(int, int, int)
MLX_MIXED_SCALAR_GETTER(float, float, float)

#undef MLX_MIXED_SCALAR_GETTER

extern "C" bool mlx_export_state_optional_float_has_value(
    const mlx_export_state state) {
  try {
    return std::get<std::optional<float>>(deref_state(state)).has_value();
  } catch (std::exception&) {
    return false;
  }
}

extern "C" int mlx_export_state_optional_float(
    float* out,
    const mlx_export_state state) {
  try {
    const auto& opt = std::get<std::optional<float>>(deref_state(state));
    if (!opt.has_value()) {
      mlx_error("mlx_export_state_optional_float called on empty optional");
      return 1;
    }
    *out = *opt;
    return 0;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
}

extern "C" const char* mlx_export_state_string(
    const mlx_export_state state) {
  return std::get<std::string>(deref_state(state)).c_str();
}

// --- Entry point ---

extern "C" int mlx_export_function_callback(
    const mlx_closure_export_callback callback,
    const mlx_closure_kwargs fun,
    const mlx_vector_array args,
    const mlx_map_string_to_array kwargs,
    bool shapeless) {
  try {
    mlx::core::export_function(
        mlx_closure_export_callback_get_(callback),
        mlx_closure_kwargs_get_(fun),
        mlx_vector_array_get_(args),
        mlx_map_string_to_array_get_(kwargs),
        shapeless);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
