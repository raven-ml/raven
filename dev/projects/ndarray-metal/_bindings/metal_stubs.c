#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/bigarray.h>  // For bigarray handling
#include <caml/callback.h>  // Might need for option processing
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <stdbool.h>
#include <string.h>

#include "metal_impl.h"

// Helper for raising Metal_error exception
static void raise_metal_error(const char* msg) {
  // Ensure the exception is registered
  static const value* exn = NULL;
  if (exn == NULL) {
    exn = caml_named_value("Metal_error");
    if (exn == NULL) {
      caml_failwith(msg);  // Fallback
      return;
    }
  }
  caml_raise_with_string(*exn, msg);
}

// Macro for checking NULL and raising error
#define CHECK_NULL(ptr, msg) \
  if ((ptr) == NULL) {       \
    raise_metal_error(msg);  \
  }

// Macro to retrieve Metal object pointer from custom block
#define MTL_PTR(v) (*(void**)Data_custom_val(v))

/* Define custom operations struct with a macro */
#define MTL_CUSTOM_OPS(name, finalize_func)         \
  static struct custom_operations name##_ops = {    \
      "ocaml.metal." #name, /* Unique identifier */ \
      finalize_func,        /* Finalizer */         \
      custom_compare_default,                       \
      custom_hash_default,                          \
      custom_serialize_default,                     \
      custom_deserialize_default,                   \
      custom_compare_ext_default,                   \
      custom_fixed_length_default}

/* Finalizer functions */
static void finalize_mtl_object(value v) { metal_release_object(MTL_PTR(v)); }

/* Custom operations for each type */
MTL_CUSTOM_OPS(device, finalize_mtl_object);
MTL_CUSTOM_OPS(command_queue, finalize_mtl_object);
MTL_CUSTOM_OPS(buffer, finalize_mtl_object);
MTL_CUSTOM_OPS(library, finalize_mtl_object);
MTL_CUSTOM_OPS(function, finalize_mtl_object);
MTL_CUSTOM_OPS(pipeline_state, finalize_mtl_object);
MTL_CUSTOM_OPS(command_buffer, finalize_mtl_object);
MTL_CUSTOM_OPS(command_encoder, finalize_mtl_object);
MTL_CUSTOM_OPS(blit_command_encoder, finalize_mtl_object);
MTL_CUSTOM_OPS(compile_options, finalize_mtl_object);

// Helper to allocate custom block
static value alloc_custom_mtl(struct custom_operations* ops, void* ptr) {
  // Assumes ptr is already retained by the metal_impl function
  CAMLparam0();  // Needed if allocation might trigger GC
  CAMLlocal1(v);
  v = caml_alloc_custom(ops, sizeof(void*), 0, 1);
  MTL_PTR(v) = ptr;
  CAMLreturn(v);
}

// **Device Management**
CAMLprim value caml_metal_create_device(value unit) {
  CAMLparam1(unit);
  MTLDevice_t device = metal_create_system_default_device();
  CHECK_NULL(device, "Could not create Metal device");
  // alloc_custom_mtl now handles CAMLparam/local/return internally
  CAMLreturn(alloc_custom_mtl(&device_ops, device));
}

CAMLprim value caml_metal_get_device_name(value device_val) {
  CAMLparam1(device_val);
  CAMLlocal1(str_val);
  MTLDevice_t device = MTL_PTR(device_val);
  NSString_t ns_str = metal_device_get_name(device);
  CHECK_NULL(ns_str, "Could not get device name");
  const char* str =
      metal_string_to_utf8(ns_str);  // Assume returns valid C string
  str_val = caml_copy_string(str);
  metal_release_object(ns_str);  // Release the NSString returned by get_name
  CAMLreturn(str_val);
}

// **Command Queue**
CAMLprim value caml_metal_new_command_queue(value device_val) {
  CAMLparam1(device_val);
  MTLDevice_t device = MTL_PTR(device_val);
  MTLCommandQueue_t queue = metal_device_new_command_queue(device);
  CHECK_NULL(queue, "Could not create command queue");
  CAMLreturn(alloc_custom_mtl(&command_queue_ops, queue));
}

// **Buffer Management**
CAMLprim value caml_metal_new_buffer(value device_val, value length_val,
                                     value options_val) {
  CAMLparam3(device_val, length_val, options_val);
  MTLDevice_t device = MTL_PTR(device_val);
  size_t length = Int64_val(length_val);
  unsigned int options = Int_val(options_val);
  MTLBuffer_t buffer = metal_device_new_buffer(device, length, options);
  CHECK_NULL(buffer, "Could not create buffer");
  CAMLreturn(alloc_custom_mtl(&buffer_ops, buffer));
}

CAMLprim value caml_metal_new_buffer_with_pointer(value device_val,
                                                  value ptr_val,
                                                  value length_val,
                                                  value options_val) {
  CAMLparam4(device_val, ptr_val, length_val, options_val);
  MTLDevice_t device = MTL_PTR(device_val);
  void* data = (void*)Nativeint_val(ptr_val);
  size_t length = Int64_val(length_val);
  unsigned int options = Int_val(options_val);
  MTLBuffer_t buffer =
      metal_device_new_buffer_with_bytes_no_copy(device, data, length, options);
  CHECK_NULL(buffer, "Could not create buffer with pointer");
  CAMLreturn(alloc_custom_mtl(&buffer_ops, buffer));
}

CAMLprim value caml_metal_new_buffer_with_data(value device_val,
                                               value bigarray_val,
                                               value options_val) {
  CAMLparam3(device_val, bigarray_val, options_val);
  MTLDevice_t device = MTL_PTR(device_val);
  struct caml_ba_array* ba = Caml_ba_array_val(bigarray_val);
  void* data = ba->data;
  size_t length = caml_ba_byte_size(ba);  // Calculate total size in bytes
  unsigned int options = Int_val(options_val);

  // Create a new buffer and copy the data
  MTLBuffer_t buffer = metal_device_new_buffer(device, length, options);
  CHECK_NULL(buffer, "Could not create buffer for bigarray");

  // Copy data to the buffer
  void* contents = metal_buffer_contents(buffer);
  CHECK_NULL(contents, "Cannot access buffer contents for bigarray copy");

  memcpy(contents, data, length);

  // If this is a managed buffer, inform Metal about the changes
  if ((options & (1 << 4)) == (1 << 4)) {  // Storage_Mode_Managed
    metal_buffer_did_modify_range(buffer, 0, length);
  }

  CAMLreturn(alloc_custom_mtl(&buffer_ops, buffer));
}

CAMLprim value caml_metal_new_buffer_with_bytes(value device_val,
                                                value bytes_val,
                                                value options_val) {
  CAMLparam3(device_val, bytes_val, options_val);
  MTLDevice_t device = MTL_PTR(device_val);
  void* data = Bytes_val(bytes_val);
  size_t length = caml_string_length(bytes_val);
  unsigned int options = Int_val(options_val);

  // Create a new buffer and copy the data
  MTLBuffer_t buffer = metal_device_new_buffer(device, length, options);
  CHECK_NULL(buffer, "Could not create buffer for bytes");

  // Copy data to the buffer
  void* contents = metal_buffer_contents(buffer);
  CHECK_NULL(contents, "Cannot access buffer contents for bytes copy");

  memcpy(contents, data, length);

  // If this is a managed buffer, inform Metal about the changes
  if ((options & (1 << 4)) == (1 << 4)) {  // Storage_Mode_Managed
    metal_buffer_did_modify_range(buffer, 0, length);
  }

  CAMLreturn(alloc_custom_mtl(&buffer_ops, buffer));
}

CAMLprim value caml_metal_buffer_length(value buffer_val) {
  CAMLparam1(buffer_val);
  MTLBuffer_t buffer = MTL_PTR(buffer_val);
  size_t length = metal_buffer_length(buffer);
  CAMLreturn(caml_copy_int64(length));
}

CAMLprim value caml_metal_buffer_contents(value buffer_val) {
  CAMLparam1(buffer_val);
  MTLBuffer_t buffer = MTL_PTR(buffer_val);
  void* contents = metal_buffer_contents(buffer);
  // The check now happens in OCaml documentation, but adding one here is safer
  CHECK_NULL(contents,
             "Cannot access contents: Buffer might be private or invalid");
  CAMLreturn(caml_copy_nativeint((intnat)contents));
}

CAMLprim value caml_metal_copy_to_buffer(value buffer_val, value offset_val,
                                         value src_ptr_val,
                                         value num_bytes_val) {
  CAMLparam4(buffer_val, offset_val, src_ptr_val, num_bytes_val);
  MTLBuffer_t buffer = MTL_PTR(buffer_val);
  size_t offset = Int64_val(offset_val);  // Use int64
  void* src = (void*)Nativeint_val(src_ptr_val);
  size_t num_bytes = Int64_val(num_bytes_val);
  void* contents = metal_buffer_contents(buffer);
  CHECK_NULL(contents,
             "Cannot copy to buffer: Storage mode might not allow CPU access");

  // Basic bounds check (optional but recommended)
  size_t buf_len = metal_buffer_length(buffer);
  if (offset + num_bytes > buf_len) {
    raise_metal_error("Buffer copy out of bounds");
  }

  char* dst = (char*)contents + offset;
  memcpy(dst, src, num_bytes);
  CAMLreturn(Val_unit);
}

CAMLprim value caml_metal_copy_from_buffer(value buffer_val, value offset_val,
                                           value dst_ptr_val,
                                           value num_bytes_val) {
  CAMLparam4(buffer_val, offset_val, dst_ptr_val, num_bytes_val);
  MTLBuffer_t buffer = MTL_PTR(buffer_val);
  size_t offset = Int64_val(offset_val);  // Use int64
  void* dst = (void*)Nativeint_val(dst_ptr_val);
  size_t num_bytes = Int64_val(num_bytes_val);
  void* contents = metal_buffer_contents(buffer);
  CHECK_NULL(
      contents,
      "Cannot copy from buffer: Storage mode might not allow CPU access");

  // Basic bounds check (optional but recommended)
  size_t buf_len = metal_buffer_length(buffer);
  if (offset + num_bytes > buf_len) {
    raise_metal_error("Buffer copy out of bounds");
  }

  char* src = (char*)contents + offset;
  memcpy(dst, src, num_bytes);
  CAMLreturn(Val_unit);
}

CAMLprim value caml_metal_buffer_did_modify_range(value buffer_val,
                                                  value offset_val,
                                                  value length_val) {
  CAMLparam3(buffer_val, offset_val, length_val);
  MTLBuffer_t buffer = MTL_PTR(buffer_val);
  size_t offset = Int64_val(offset_val);
  size_t length = Int64_val(length_val);
  metal_buffer_did_modify_range(buffer, offset, length);
  CAMLreturn(Val_unit);
}

// **Compile Options**
CAMLprim value caml_metal_create_compile_options(value unit) {
  CAMLparam1(unit);
  MTLCompileOptions_t options = metal_create_compile_options();
  CHECK_NULL(options, "Could not create compile options");
  CAMLreturn(alloc_custom_mtl(&compile_options_ops, options));
}

CAMLprim value caml_metal_set_compile_option_fast_math_enabled(
    value options_val, value enabled_val) {
  CAMLparam2(options_val, enabled_val);
  MTLCompileOptions_t options = MTL_PTR(options_val);
  bool enabled = Bool_val(enabled_val);
  metal_compile_options_set_fast_math_enabled(options, enabled);
  CAMLreturn(Val_unit);
}

// **Library Management**
CAMLprim value caml_metal_new_library_with_source(value device_val,
                                                  value source_val,
                                                  value options_val_opt) {
  CAMLparam3(device_val, source_val, options_val_opt);
  CAMLlocal1(library_val);  // Local value needed for alloc_custom_mtl result
  MTLDevice_t device = MTL_PTR(device_val);
  const char* source = String_val(source_val);
  MTLCompileOptions_t options_ptr = NULL;

  // Use Is_block to check for Some(v) vs None (Val_int(0))
  if (Is_block(options_val_opt)) {
    options_ptr = MTL_PTR(Field(options_val_opt, 0));
  }

  MTLLibrary_t library =
      metal_device_new_library_with_source(device, source, options_ptr);
  // TODO: Handle potential NSError** from the underlying implementation
  CHECK_NULL(library,
             "Could not create library from source (check compiler errors)");

  library_val = alloc_custom_mtl(&library_ops, library);
  CAMLreturn(library_val);
}

CAMLprim value caml_metal_new_library_with_data(value device_val,
                                                value data_val) {
  CAMLparam2(device_val, data_val);
  CAMLlocal1(library_val);  // Local value needed for alloc_custom_mtl result
  MTLDevice_t device = MTL_PTR(device_val);
  // Using Bytes_val might be safer if data isn't guaranteed
  // UTF-8/null-terminated but String_val gives a char* directly. Let's stick
  // with String_val for now.
  const char* data = String_val(data_val);
  size_t length = caml_string_length(data_val);  // Use length from OCaml value
  MTLLibrary_t library =
      metal_device_new_library_with_data(device, data, length);
  // TODO: Check for NSError** like in source version
  CHECK_NULL(library, "Could not create library from data");

  library_val = alloc_custom_mtl(&library_ops, library);
  CAMLreturn(library_val);
}

// **Function Management**
CAMLprim value caml_metal_new_function_with_name(value library_val,
                                                 value name_val) {
  CAMLparam2(library_val, name_val);
  CAMLlocal1(function_val);
  MTLLibrary_t library = MTL_PTR(library_val);
  const char* name = String_val(name_val);
  MTLFunction_t function = metal_library_new_function_with_name(library, name);
  char error_msg[100];
  sprintf(error_msg,
          "Could not create function with name '%s' (check compiler errors)",
          name);
  CHECK_NULL(function, error_msg);
  function_val = alloc_custom_mtl(&function_ops, function);
  CAMLreturn(function_val);
}

// **Pipeline State Management**
CAMLprim value caml_metal_new_compute_pipeline_state(value device_val,
                                                     value function_val) {
  CAMLparam2(device_val, function_val);
  CAMLlocal1(pipeline_state_val);
  MTLDevice_t device = MTL_PTR(device_val);
  MTLFunction_t function = MTL_PTR(function_val);
  // TODO: Underlying ObjC returns NSError**. Update impl.
  MTLComputePipelineState_t pipeline_state =
      metal_device_new_compute_pipeline_state_with_function(device, function);
  CHECK_NULL(pipeline_state, "Could not create compute pipeline state");
  pipeline_state_val = alloc_custom_mtl(&pipeline_state_ops, pipeline_state);
  CAMLreturn(pipeline_state_val);
}

CAMLprim value caml_metal_pipeline_state_get_max_total_threads_per_threadgroup(
    value pipeline_state_val) {
  CAMLparam1(pipeline_state_val);
  MTLComputePipelineState_t pipeline_state = MTL_PTR(pipeline_state_val);
  size_t max_threads =
      metal_pipeline_state_get_max_total_threads_per_threadgroup(
          pipeline_state);
  CAMLreturn(caml_copy_int64(max_threads));  // Return int64
}

// **Command Buffer Management**
CAMLprim value caml_metal_new_command_buffer(value queue_val) {
  CAMLparam1(queue_val);
  CAMLlocal1(command_buffer_val);
  MTLCommandQueue_t queue = MTL_PTR(queue_val);
  MTLCommandBuffer_t command_buffer = metal_queue_new_command_buffer(queue);
  CHECK_NULL(command_buffer, "Could not create command buffer");
  command_buffer_val = alloc_custom_mtl(&command_buffer_ops, command_buffer);
  CAMLreturn(command_buffer_val);
}

CAMLprim value caml_metal_commit(value command_buffer_val) {
  CAMLparam1(command_buffer_val);
  MTLCommandBuffer_t command_buffer = MTL_PTR(command_buffer_val);
  metal_command_buffer_commit(command_buffer);
  CAMLreturn(Val_unit);
}

CAMLprim value caml_metal_wait_until_completed(value command_buffer_val) {
  CAMLparam1(command_buffer_val);
  MTLCommandBuffer_t command_buffer = MTL_PTR(command_buffer_val);
  metal_command_buffer_wait_until_completed(command_buffer);
  CAMLreturn(Val_unit);
}

// **Compute Command Encoder Management**
CAMLprim value
caml_metal_new_compute_command_encoder(value command_buffer_val) {
  CAMLparam1(command_buffer_val);
  CAMLlocal1(encoder_val);
  MTLCommandBuffer_t command_buffer = MTL_PTR(command_buffer_val);
  MTLComputeCommandEncoder_t encoder =
      metal_command_buffer_compute_command_encoder(command_buffer);
  CHECK_NULL(encoder, "Could not create compute command encoder");
  encoder_val = alloc_custom_mtl(&command_encoder_ops, encoder);
  CAMLreturn(encoder_val);
}

CAMLprim value caml_metal_set_compute_pipeline_state(value encoder_val,
                                                     value pipeline_state_val) {
  CAMLparam2(encoder_val, pipeline_state_val);
  MTLComputeCommandEncoder_t encoder = MTL_PTR(encoder_val);
  MTLComputePipelineState_t pipeline_state = MTL_PTR(pipeline_state_val);
  metal_compute_command_encoder_set_pipeline_state(encoder, pipeline_state);
  CAMLreturn(Val_unit);
}

CAMLprim value caml_metal_set_buffer(value encoder_val, value buffer_val,
                                     value offset_val, value index_val) {
  CAMLparam4(encoder_val, buffer_val, offset_val, index_val);
  MTLComputeCommandEncoder_t encoder = MTL_PTR(encoder_val);
  MTLBuffer_t buffer = MTL_PTR(buffer_val);
  size_t offset = Int64_val(offset_val);  // Use int64 -> size_t
  uint32_t index = Int_val(index_val);
  metal_compute_command_encoder_set_buffer(encoder, buffer, offset, index);
  CAMLreturn(Val_unit);
}

CAMLprim value caml_metal_set_bytes(value encoder_val, value bytes_ptr_val,
                                    value length_val, value index_val) {
  CAMLparam4(encoder_val, bytes_ptr_val, length_val, index_val);
  MTLComputeCommandEncoder_t encoder = MTL_PTR(encoder_val);
  void* bytes_ptr = (void*)Nativeint_val(bytes_ptr_val);
  size_t length =
      Int64_val(length_val);  // Use Int64_val for size_t consistency
  uint32_t index = Int_val(index_val);
  metal_compute_command_encoder_set_bytes(encoder, bytes_ptr, length, index);
  CAMLreturn(Val_unit);
}

CAMLprim value caml_metal_dispatch_thread_groups(value encoder_val,
                                                 value grid_val,
                                                 value thread_val) {
  CAMLparam3(encoder_val, grid_val, thread_val);
  MTLComputeCommandEncoder_t encoder = MTL_PTR(encoder_val);
  // Grid/thread sizes are counts, size_t is appropriate
  size_t grid_x = Int64_val(Field(grid_val, 0));
  size_t grid_y = Int64_val(Field(grid_val, 1));
  size_t grid_z = Int64_val(Field(grid_val, 2));
  size_t thread_x = Int64_val(Field(thread_val, 0));
  size_t thread_y = Int64_val(Field(thread_val, 1));
  size_t thread_z = Int64_val(Field(thread_val, 2));
  metal_compute_command_encoder_dispatch_thread_groups(
      encoder, grid_x, grid_y, grid_z, thread_x, thread_y, thread_z);
  CAMLreturn(Val_unit);
}

CAMLprim value caml_metal_end_encoding(value encoder_val) {
  CAMLparam1(encoder_val);
  MTLComputeCommandEncoder_t encoder = MTL_PTR(encoder_val);
  metal_compute_command_encoder_end_encoding(encoder);
  CAMLreturn(Val_unit);
}

// **Blit Command Encoder Management**
CAMLprim value caml_metal_new_blit_command_encoder(value command_buffer_val) {
  CAMLparam1(command_buffer_val);
  CAMLlocal1(encoder_val);
  MTLCommandBuffer_t command_buffer = MTL_PTR(command_buffer_val);
  MTLBlitCommandEncoder_t encoder =
      metal_command_buffer_blit_command_encoder(command_buffer);
  CHECK_NULL(encoder, "Could not create blit command encoder");
  encoder_val = alloc_custom_mtl(&blit_command_encoder_ops, encoder);
  CAMLreturn(encoder_val);
}

CAMLprim value caml_metal_blit_copy_from_buffer_to_buffer(
    value encoder_val, value src_buffer_val, value src_offset_val,
    value dst_buffer_val, value dst_offset_val, value size_val) {
  // Correct usage for 6 parameters
  CAMLparam5(encoder_val, src_buffer_val, src_offset_val, dst_buffer_val,
             dst_offset_val);
  CAMLxparam1(size_val);
  MTLBlitCommandEncoder_t encoder = MTL_PTR(encoder_val);
  MTLBuffer_t src_buffer = MTL_PTR(src_buffer_val);
  size_t src_offset = Int64_val(src_offset_val);  // Use int64
  MTLBuffer_t dst_buffer = MTL_PTR(dst_buffer_val);
  size_t dst_offset = Int64_val(dst_offset_val);  // Use int64
  size_t size = Int64_val(size_val);
  metal_blit_command_encoder_copy_from_buffer(encoder, src_buffer, src_offset,
                                              dst_buffer, dst_offset, size);
  CAMLreturn(Val_unit);
}

// Bytecode stub for copy_from_buffer_to_buffer (6 args)
CAMLprim value caml_metal_blit_copy_from_buffer_to_buffer_bytecode(value* argv,
                                                                   int argn) {
  return caml_metal_blit_copy_from_buffer_to_buffer(argv[0], argv[1], argv[2],
                                                    argv[3], argv[4], argv[5]);
}

CAMLprim value caml_metal_end_blit_encoding(value encoder_val) {
  CAMLparam1(encoder_val);
  MTLBlitCommandEncoder_t encoder = MTL_PTR(encoder_val);
  metal_blit_command_encoder_end_encoding(encoder);
  CAMLreturn(Val_unit);
}
