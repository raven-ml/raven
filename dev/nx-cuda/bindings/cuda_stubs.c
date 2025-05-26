#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <cuda.h>

/* Initialization */
value caml_cu_init(value flags_val) {
  CAMLparam1(flags_val);
  unsigned int flags = Int_val(flags_val);
  CUresult res = cuInit(flags);
  if (res != CUDA_SUCCESS) caml_failwith("cuInit failed");
  CAMLreturn(Val_unit);
}

/* Device Management */
value caml_cu_device_get_count(value unit) {
  CAMLparam1(unit);
  int count;
  CUresult res = cuDeviceGetCount(&count);
  if (res != CUDA_SUCCESS) caml_failwith("cuDeviceGetCount failed");
  CAMLreturn(Val_int(count));
}

value caml_cu_device_get(value ordinal_val) {
  CAMLparam1(ordinal_val);
  int ordinal = Int_val(ordinal_val);
  CUdevice device;
  CUresult res = cuDeviceGet(&device, ordinal);
  if (res != CUDA_SUCCESS) caml_failwith("cuDeviceGet failed");
  CAMLreturn(Val_int(device));
}

/* Context Management */
value caml_cu_ctx_create(value flags_val, value dev_val) {
  CAMLparam2(flags_val, dev_val);
  unsigned int flags = Int_val(flags_val);
  CUdevice dev = Int_val(dev_val);
  CUcontext ctx;
  CUresult res = cuCtxCreate(&ctx, flags, dev);
  if (res != CUDA_SUCCESS) caml_failwith("cuCtxCreate failed");
  CAMLreturn(caml_copy_int64((int64_t)(uintptr_t)ctx));
}

value caml_cu_ctx_get_flags(value unit) {
  CAMLparam1(unit);
  unsigned int flags;
  CUresult res = cuCtxGetFlags(&flags);
  if (res != CUDA_SUCCESS) caml_failwith("cuCtxGetFlags failed");
  CAMLreturn(Val_int(flags));
}

value caml_cu_device_primary_ctx_retain(value dev_val) {
  CAMLparam1(dev_val);
  CUdevice dev = Int_val(dev_val);
  CUcontext ctx;
  CUresult res = cuDevicePrimaryCtxRetain(&ctx, dev);
  if (res != CUDA_SUCCESS) caml_failwith("cuDevicePrimaryCtxRetain failed");
  CAMLreturn(caml_copy_int64((int64_t)(uintptr_t)ctx));
}

value caml_cu_device_primary_ctx_release(value dev_val) {
  CAMLparam1(dev_val);
  CUdevice dev = Int_val(dev_val);
  CUresult res = cuDevicePrimaryCtxRelease(dev);
  if (res != CUDA_SUCCESS) caml_failwith("cuDevicePrimaryCtxRelease failed");
  CAMLreturn(Val_unit);
}

value caml_cu_device_primary_ctx_reset(value dev_val) {
  CAMLparam1(dev_val);
  CUdevice dev = Int_val(dev_val);
  CUresult res = cuDevicePrimaryCtxReset(dev);
  if (res != CUDA_SUCCESS) caml_failwith("cuDevicePrimaryCtxReset failed");
  CAMLreturn(Val_unit);
}

value caml_cu_ctx_get_device(value unit) {
  CAMLparam1(unit);
  CUdevice device;
  CUresult res = cuCtxGetDevice(&device);
  if (res != CUDA_SUCCESS) caml_failwith("cuCtxGetDevice failed");
  CAMLreturn(Val_int(device));
}

value caml_cu_ctx_get_current(value unit) {
  CAMLparam1(unit);
  CUcontext ctx;
  CUresult res = cuCtxGetCurrent(&ctx);
  if (res != CUDA_SUCCESS) caml_failwith("cuCtxGetCurrent failed");
  CAMLreturn(caml_copy_int64((int64_t)(uintptr_t)ctx));
}

value caml_cu_ctx_pop_current(value unit) {
  CAMLparam1(unit);
  CUcontext ctx;
  CUresult res = cuCtxPopCurrent(&ctx);
  if (res != CUDA_SUCCESS) caml_failwith("cuCtxPopCurrent failed");
  CAMLreturn(caml_copy_int64((int64_t)(uintptr_t)ctx));
}

value caml_cu_ctx_set_current(value ctx_val) {
  CAMLparam1(ctx_val);
  CUcontext ctx = (CUcontext)(uintptr_t)Int64_val(ctx_val);
  CUresult res = cuCtxSetCurrent(ctx);
  if (res != CUDA_SUCCESS) caml_failwith("cuCtxSetCurrent failed");
  CAMLreturn(Val_unit);
}

value caml_cu_ctx_push_current(value ctx_val) {
  CAMLparam1(ctx_val);
  CUcontext ctx = (CUcontext)(uintptr_t)Int64_val(ctx_val);
  CUresult res = cuCtxPushCurrent(ctx);
  if (res != CUDA_SUCCESS) caml_failwith("cuCtxPushCurrent failed");
  CAMLreturn(Val_unit);
}

/* Module Management */
value caml_cu_module_load_data_ex(value image_val, value options_list) {
  CAMLparam2(image_val, options_list);
  const char* image = String_val(image_val);
  int numOptions = Wosize_val(options_list);
  CUjit_option* options = NULL;
  void** optionValues = NULL;
  if (numOptions > 0) {
    options = (CUjit_option*)malloc(numOptions * sizeof(CUjit_option));
    optionValues = (void**)malloc(numOptions * sizeof(void*));
    for (int i = 0; i < numOptions; i++) {
      value pair = Field(options_list, i);
      options[i] = (CUjit_option)Int_val(Field(pair, 0));
      optionValues[i] = (void*)(uintptr_t)Int64_val(Field(pair, 1));
    }
  }
  CUmodule mod;
  CUresult res =
      cuModuleLoadDataEx(&mod, image, numOptions, options, optionValues);
  free(options);
  free(optionValues);
  if (res != CUDA_SUCCESS) {
    const char* error_string;
    cuGetErrorString(res, &error_string);
    char error_msg[256];
    snprintf(error_msg, sizeof(error_msg), "cuModuleLoadDataEx failed: %s",
             error_string);
    caml_failwith(error_msg);
  }

  CAMLreturn(caml_copy_int64((int64_t)(uintptr_t)mod));
}

value caml_cu_module_get_function(value mod_val, value name_val) {
  CAMLparam2(mod_val, name_val);
  CUmodule mod = (CUmodule)(uintptr_t)Int64_val(mod_val);
  CUfunction func;
  CUresult res = cuModuleGetFunction(&func, mod, String_val(name_val));
  if (res != CUDA_SUCCESS) {
    const char* error_string;
    cuGetErrorString(res, &error_string);
    char error_msg[256];
    snprintf(error_msg, sizeof(error_msg), "cuModuleGetFunction failed: %s",
             error_string);
    caml_failwith(error_msg);
  }
  CAMLreturn(caml_copy_int64((int64_t)(uintptr_t)func));
}

/* Memory Management */
value caml_cu_mem_alloc(value bytesize_val) {
  CAMLparam1(bytesize_val);
  size_t bytesize = (size_t)Int64_val(bytesize_val);
  CUdeviceptr dptr;
  CUresult res = cuMemAlloc(&dptr, bytesize);
  if (res != CUDA_SUCCESS) caml_failwith("cuMemAlloc failed");
  CAMLreturn(caml_copy_int64((int64_t)dptr));
}

value caml_cu_mem_alloc_async(value bytesize_val, value stream_val) {
  CAMLparam2(bytesize_val, stream_val);
  size_t bytesize = (size_t)Int64_val(bytesize_val);
  CUstream stream = (CUstream)(uintptr_t)Int64_val(stream_val);
  CUdeviceptr dptr;
  CUresult res = cuMemAllocAsync(&dptr, bytesize, stream);
  if (res != CUDA_SUCCESS) caml_failwith("cuMemAllocAsync failed");
  CAMLreturn(caml_copy_int64((int64_t)dptr));
}

value caml_cu_memcpy_H_to_D(value dst_val, value src_bytes_val,
                            value count_val) {
  CAMLparam3(dst_val, src_bytes_val, count_val);
  CUdeviceptr dst = (CUdeviceptr)Int64_val(dst_val);
  size_t count = (size_t)Int64_val(count_val);
  if (caml_string_length(src_bytes_val) < count) {
    caml_failwith("bytes value too small for cuMemcpyHtoD");
  }
  void* src = (void*)String_val(src_bytes_val);
  CUresult res = cuMemcpyHtoD(dst, src, count);
  if (res != CUDA_SUCCESS) caml_failwith("cuMemcpyHtoD failed");
  CAMLreturn(Val_unit);
}

value caml_cu_memcpy_H_to_D_async(value dst_val, value src_bytes_val,
                                  value count_val, value stream_val) {
  CAMLparam4(dst_val, src_bytes_val, count_val, stream_val);
  CUdeviceptr dst = (CUdeviceptr)Int64_val(dst_val);
  size_t count = (size_t)Int64_val(count_val);
  if (caml_string_length(src_bytes_val) < count) {
    caml_failwith("bytes value too small for cuMemcpyHtoDAsync");
  }
  void* src = (void*)String_val(src_bytes_val);
  CUstream stream = (CUstream)(uintptr_t)Int64_val(stream_val);
  CUresult res = cuMemcpyHtoDAsync(dst, src, count, stream);
  if (res != CUDA_SUCCESS) caml_failwith("cuMemcpyHtoDAsync failed");
  CAMLreturn(Val_unit);
}

value caml_cu_memcpy_D_to_H(value dst_bytes_val, value src_val,
                            value count_val) {
  CAMLparam3(dst_bytes_val, src_val, count_val);
  size_t count = (size_t)Int64_val(count_val);
  if (caml_string_length(dst_bytes_val) < count) {
    caml_failwith("bytes value too small for cuMemcpyDtoH");
  }
  void* dst = (void*)String_val(dst_bytes_val);
  CUdeviceptr src = (CUdeviceptr)Int64_val(src_val);
  CUresult res = cuMemcpyDtoH(dst, src, count);
  if (res != CUDA_SUCCESS) caml_failwith("cuMemcpyDtoH failed");
  CAMLreturn(Val_unit);
}

value caml_cu_memcpy_D_to_H_async(value dst_bytes_val, value src_val,
                                  value count_val, value stream_val) {
  CAMLparam4(dst_bytes_val, src_val, count_val, stream_val);
  size_t count = (size_t)Int64_val(count_val);
  if (caml_string_length(dst_bytes_val) < count) {
    caml_failwith("bytes value too small for cuMemcpyDtoHAsync");
  }
  void* dst = (void*)String_val(dst_bytes_val);
  CUdeviceptr src = (CUdeviceptr)Int64_val(src_val);
  CUstream stream = (CUstream)(uintptr_t)Int64_val(stream_val);
  CUresult res = cuMemcpyDtoHAsync(dst, src, count, stream);
  if (res != CUDA_SUCCESS) caml_failwith("cuMemcpyDtoHAsync failed");
  CAMLreturn(Val_unit);
}

value caml_cu_memcpy_D_to_D(value dst_val, value src_val, value count_val) {
  CAMLparam3(dst_val, src_val, count_val);
  CUdeviceptr dst = (CUdeviceptr)Int64_val(dst_val);
  CUdeviceptr src = (CUdeviceptr)Int64_val(src_val);
  size_t count = (size_t)Int64_val(count_val);
  CUresult res = cuMemcpyDtoD(dst, src, count);
  if (res != CUDA_SUCCESS) caml_failwith("cuMemcpyDtoD failed");
  CAMLreturn(Val_unit);
}

value caml_cu_memcpy_D_to_D_async(value dst_val, value src_val, value count_val,
                                  value stream_val) {
  CAMLparam4(dst_val, src_val, count_val, stream_val);
  CUdeviceptr dst = (CUdeviceptr)Int64_val(dst_val);
  CUdeviceptr src = (CUdeviceptr)Int64_val(src_val);
  size_t count = (size_t)Int64_val(count_val);
  CUstream stream = (CUstream)(uintptr_t)Int64_val(stream_val);
  CUresult res = cuMemcpyDtoDAsync(dst, src, count, stream);
  if (res != CUDA_SUCCESS) caml_failwith("cuMemcpyDtoDAsync failed");
  CAMLreturn(Val_unit);
}

value caml_cu_memcpy_peer(value dst_val, value dst_ctx_val, value src_val,
                          value src_ctx_val, value count_val) {
  CAMLparam5(dst_val, dst_ctx_val, src_val, src_ctx_val, count_val);
  CUdeviceptr dst = (CUdeviceptr)Int64_val(dst_val);
  CUcontext dst_ctx = (CUcontext)(uintptr_t)Int64_val(dst_ctx_val);
  CUdeviceptr src = (CUdeviceptr)Int64_val(src_val);
  CUcontext src_ctx = (CUcontext)(uintptr_t)Int64_val(src_ctx_val);
  size_t count = (size_t)Int64_val(count_val);
  CUresult res = cuMemcpyPeer(dst, dst_ctx, src, src_ctx, count);
  if (res != CUDA_SUCCESS) caml_failwith("cuMemcpyPeer failed");
  CAMLreturn(Val_unit);
}

value caml_cu_memcpy_peer_async(value dst_val, value dst_ctx_val, value src_val,
                                value src_ctx_val, value count_val,
                                value stream_val) {
  CAMLparam5(dst_val, dst_ctx_val, src_val, src_ctx_val, count_val);
  CAMLxparam1(stream_val);
  CUdeviceptr dst = (CUdeviceptr)Int64_val(dst_val);
  CUcontext dst_ctx = (CUcontext)(uintptr_t)Int64_val(dst_ctx_val);
  CUdeviceptr src = (CUdeviceptr)Int64_val(src_val);
  CUcontext src_ctx = (CUcontext)(uintptr_t)Int64_val(src_ctx_val);
  size_t count = (size_t)Int64_val(count_val);
  CUstream stream = (CUstream)(uintptr_t)Int64_val(stream_val);
  CUresult res = cuMemcpyPeerAsync(dst, dst_ctx, src, src_ctx, count, stream);
  if (res != CUDA_SUCCESS) caml_failwith("cuMemcpyPeerAsync failed");
  CAMLreturn(Val_unit);
}

value caml_cu_memcpy_peer_async_bytecode(value* argv, int argn) {
  return caml_cu_memcpy_peer_async(argv[0], argv[1], argv[2], argv[3], argv[4],
                                   argv[5]);
}

value caml_cu_mem_free(value dptr_val) {
  CAMLparam1(dptr_val);
  CUdeviceptr dptr = (CUdeviceptr)Int64_val(dptr_val);
  CUresult res = cuMemFree(dptr);
  if (res != CUDA_SUCCESS) caml_failwith("cuMemFree failed");
  CAMLreturn(Val_unit);
}

value caml_cu_mem_free_async(value dptr_val, value stream_val) {
  CAMLparam2(dptr_val, stream_val);
  CUdeviceptr dptr = (CUdeviceptr)Int64_val(dptr_val);
  CUstream stream = (CUstream)(uintptr_t)Int64_val(stream_val);
  CUresult res = cuMemFreeAsync(dptr, stream);
  if (res != CUDA_SUCCESS) caml_failwith("cuMemFreeAsync failed");
  CAMLreturn(Val_unit);
}

/* Kernel Launch */
CAMLprim value caml_cu_launch_kernel(value func_val, value gridX_val,
                                     value gridY_val, value gridZ_val,
                                     value blockX_val, value blockY_val,
                                     value blockZ_val, value args_val) {
  CAMLparam5(func_val, gridX_val, gridY_val, gridZ_val, args_val);
  CAMLxparam3(blockX_val, blockY_val, blockZ_val);

  CUfunction func = (CUfunction)(uintptr_t)Int64_val(func_val);
  if (func == NULL) caml_failwith("Invalid CUfunction handle");

  unsigned int gridX = (unsigned int)Int_val(gridX_val);
  unsigned int gridY = (unsigned int)Int_val(gridY_val);
  unsigned int gridZ = (unsigned int)Int_val(gridZ_val);
  unsigned int blockX = (unsigned int)Int_val(blockX_val);
  unsigned int blockY = (unsigned int)Int_val(blockY_val);
  unsigned int blockZ = (unsigned int)Int_val(blockZ_val);

  int numArgs = Wosize_val(args_val);
  void* param_values[numArgs];
  for (int i = 0; i < numArgs; i++) {
    void* ptr = (void*)(uintptr_t)Int64_val(Field(args_val, i));
    if (ptr == NULL) {
      caml_failwith("NULL device pointer in args");
    }
    param_values[i] = ptr;
  }
  void** args = malloc(numArgs * sizeof(void*));
  if (args == NULL) caml_failwith("malloc failed");
  for (int i = 0; i < numArgs; i++) {
    args[i] = &param_values[i];
  }

  CUresult res = cuLaunchKernel(func, gridX, gridY, gridZ, blockX, blockY,
                                blockZ, 0, 0, args, 0);
  if (res != CUDA_SUCCESS) {
    const char* err_str;
    cuGetErrorString(res, &err_str);
    free(args);
    char buf[256];
    snprintf(buf, sizeof(buf), "cuLaunchKernel failed: %s", err_str);
    caml_failwith(buf);
  }

  free(args);
  CAMLreturn(Val_unit);
}

value caml_cu_launch_kernel_bytecode(value* argv, int argn) {
  return caml_cu_launch_kernel(argv[0], argv[1], argv[2], argv[3], argv[4],
                               argv[5], argv[6], argv[7]);
}

/* Synchronization */
value caml_cu_ctx_synchronize(value unit) {
  CAMLparam1(unit);
  CUresult res = cuCtxSynchronize();
  if (res != CUDA_SUCCESS) caml_failwith("cuCtxSynchronize failed");
  CAMLreturn(Val_unit);
}

/* Peer Access */
value caml_cu_ctx_disable_peer_access(value peer_ctx_val) {
  CAMLparam1(peer_ctx_val);
  CUcontext peer_ctx = (CUcontext)(uintptr_t)Int64_val(peer_ctx_val);
  CUresult res = cuCtxDisablePeerAccess(peer_ctx);
  if (res != CUDA_SUCCESS) caml_failwith("cuCtxDisablePeerAccess failed");
  CAMLreturn(Val_unit);
}

value caml_cu_ctx_enable_peer_access(value peer_ctx_val, value flags_val) {
  CAMLparam2(peer_ctx_val, flags_val);
  CUcontext peer_ctx = (CUcontext)(uintptr_t)Int64_val(peer_ctx_val);
  unsigned int flags = Int_val(flags_val);
  CUresult res = cuCtxEnablePeerAccess(peer_ctx, flags);
  if (res != CUDA_SUCCESS) caml_failwith("cuCtxEnablePeerAccess failed");
  CAMLreturn(Val_unit);
}

value caml_cu_device_can_access_peer(value dev_val, value peer_dev_val) {
  CAMLparam2(dev_val, peer_dev_val);
  CUdevice dev = Int_val(dev_val);
  CUdevice peer_dev = Int_val(peer_dev_val);
  int can_access;
  CUresult res = cuDeviceCanAccessPeer(&can_access, dev, peer_dev);
  if (res != CUDA_SUCCESS) caml_failwith("cuDeviceCanAccessPeer failed");
  CAMLreturn(Val_int(can_access));
}

value caml_cu_device_get_p2p_attribute(value attr_val, value src_dev_val,
                                       value dst_dev_val) {
  CAMLparam3(attr_val, src_dev_val, dst_dev_val);
  CUdevice_P2PAttribute attr = (CUdevice_P2PAttribute)Int_val(attr_val);
  CUdevice src_dev = Int_val(src_dev_val);
  CUdevice dst_dev = Int_val(dst_dev_val);
  int attr_value;
  CUresult res = cuDeviceGetP2PAttribute(&attr_value, attr, src_dev, dst_dev);
  if (res != CUDA_SUCCESS) caml_failwith("cuDeviceGetP2PAttribute failed");
  CAMLreturn(Val_int(attr_value));
}

/* Module Unload */
value caml_cu_module_unload(value mod_val) {
  CAMLparam1(mod_val);
  CUmodule mod = (CUmodule)(uintptr_t)Int64_val(mod_val);
  CUresult res = cuModuleUnload(mod);
  if (res != CUDA_SUCCESS) caml_failwith("cuModuleUnload failed");
  CAMLreturn(Val_unit);
}

/* Context Destruction */
value caml_cu_ctx_destroy(value ctx_val) {
  CAMLparam1(ctx_val);
  CUcontext ctx = (CUcontext)(uintptr_t)Int64_val(ctx_val);
  CUresult res = cuCtxDestroy(ctx);
  if (res != CUDA_SUCCESS) caml_failwith("cuCtxDestroy failed");
  CAMLreturn(Val_unit);
}

/* Memory Set */
value caml_cu_memset_d8(value dptr_val, value value_val, value count_val) {
  CAMLparam3(dptr_val, value_val, count_val);
  CUdeviceptr dptr = (CUdeviceptr)Int64_val(dptr_val);
  unsigned char byte_value = Int_val(value_val);
  size_t count = (size_t)Int64_val(count_val);
  CUresult res = cuMemsetD8(dptr, byte_value, count);
  if (res != CUDA_SUCCESS) caml_failwith("cuMemsetD8 failed");
  CAMLreturn(Val_unit);
}

value caml_cu_memset_d16(value dptr_val, value value_val, value count_val) {
  CAMLparam3(dptr_val, value_val, count_val);
  CUdeviceptr dptr = (CUdeviceptr)Int64_val(dptr_val);
  unsigned short short_value = Int_val(value_val);
  size_t count = (size_t)Int64_val(count_val);
  CUresult res = cuMemsetD16(dptr, short_value, count);
  if (res != CUDA_SUCCESS) caml_failwith("cuMemsetD16 failed");
  CAMLreturn(Val_unit);
}

value caml_cu_memset_d32(value dptr_val, value value_val, value count_val) {
  CAMLparam3(dptr_val, value_val, count_val);
  CUdeviceptr dptr = (CUdeviceptr)Int64_val(dptr_val);
  unsigned int int_value =
      Int_val(value_val);  // Renamed from 'value' to 'int_value'
  size_t count = (size_t)Int64_val(count_val);
  CUresult res = cuMemsetD32(dptr, int_value, count);
  if (res != CUDA_SUCCESS) caml_failwith("cuMemsetD32 failed");
  CAMLreturn(Val_unit);
}

value caml_cu_memset_d8_async(value dptr_val, value value_val, value count_val,
                              value stream_val) {
  CAMLparam4(dptr_val, value_val, count_val, stream_val);
  CUdeviceptr dptr = (CUdeviceptr)Int64_val(dptr_val);
  unsigned char byte_value =
      Int_val(value_val);  // Renamed from 'value' to 'byte_value'
  size_t count = (size_t)Int64_val(count_val);
  CUstream stream = (CUstream)(uintptr_t)Int64_val(stream_val);
  CUresult res = cuMemsetD8Async(dptr, byte_value, count, stream);
  if (res != CUDA_SUCCESS) caml_failwith("cuMemsetD8Async failed");
  CAMLreturn(Val_unit);
}

value caml_cu_memset_d16_async(value dptr_val, value value_val, value count_val,
                               value stream_val) {
  CAMLparam4(dptr_val, value_val, count_val, stream_val);
  CUdeviceptr dptr = (CUdeviceptr)Int64_val(dptr_val);
  unsigned short short_value =
      Int_val(value_val);  // Renamed from 'value' to 'short_value'
  size_t count = (size_t)Int64_val(count_val);
  CUstream stream = (CUstream)(uintptr_t)Int64_val(stream_val);
  CUresult res = cuMemsetD16Async(dptr, short_value, count, stream);
  if (res != CUDA_SUCCESS) caml_failwith("cuMemsetD16Async failed");
  CAMLreturn(Val_unit);
}

value caml_cu_memset_d32_async(value dptr_val, value value_val, value count_val,
                               value stream_val) {
  CAMLparam4(dptr_val, value_val, count_val, stream_val);
  CUdeviceptr dptr = (CUdeviceptr)Int64_val(dptr_val);
  unsigned int int_value =
      Int_val(value_val);  // Renamed from 'value' to 'int_value'
  size_t count = (size_t)Int64_val(count_val);
  CUstream stream = (CUstream)(uintptr_t)Int64_val(stream_val);
  CUresult res = cuMemsetD32Async(dptr, int_value, count, stream);
  if (res != CUDA_SUCCESS) caml_failwith("cuMemsetD32Async failed");
  CAMLreturn(Val_unit);
}

/* Memory Info */
value caml_cu_mem_get_info(value unit) {
  CAMLparam1(unit);
  size_t free, total;
  CUresult res = cuMemGetInfo(&free, &total);
  if (res != CUDA_SUCCESS) caml_failwith("cuMemGetInfo failed");
  value ret = caml_alloc_tuple(2);
  Store_field(ret, 0, caml_copy_int64((int64_t)free));
  Store_field(ret, 1, caml_copy_int64((int64_t)total));
  CAMLreturn(ret);
}

/* Module Globals */
value caml_cu_module_get_global(value mod_val, value name_val) {
  CAMLparam2(mod_val, name_val);
  CUmodule mod = (CUmodule)(uintptr_t)Int64_val(mod_val);
  CUdeviceptr dptr;
  size_t bytes;
  CUresult res = cuModuleGetGlobal(&dptr, &bytes, mod, String_val(name_val));
  if (res != CUDA_SUCCESS) caml_failwith("cuModuleGetGlobal failed");
  value ret = caml_alloc_tuple(2);
  Store_field(ret, 0, caml_copy_int64((int64_t)dptr));
  Store_field(ret, 1, caml_copy_int64((int64_t)bytes));
  CAMLreturn(ret);
}

/* Device Info */
value caml_cu_device_get_name(value len_val, value dev_val) {
  CAMLparam2(len_val, dev_val);
  int len = Int_val(len_val);
  CUdevice dev = Int_val(dev_val);
  char* name = (char*)malloc(len * sizeof(char));
  CUresult res = cuDeviceGetName(name, len, dev);
  if (res != CUDA_SUCCESS) {
    free(name);
    caml_failwith("cuDeviceGetName failed");
  }
  value ret = caml_copy_string(name);
  free(name);
  CAMLreturn(ret);
}

value caml_cu_device_get_attribute(value attr_val, value dev_val) {
  CAMLparam2(attr_val, dev_val);
  CUdevice_attribute attr = (CUdevice_attribute)Int_val(attr_val);
  CUdevice dev = Int_val(dev_val);
  int attr_value;  // Renamed from 'value' to 'attr_value'
  CUresult res = cuDeviceGetAttribute(&attr_value, attr, dev);
  if (res != CUDA_SUCCESS) caml_failwith("cuDeviceGetAttribute failed");
  CAMLreturn(Val_int(attr_value));
}

/* Context Limits */
value caml_cu_ctx_set_limit(value limit_val, value value_val) {
  CAMLparam2(limit_val, value_val);
  CUlimit limit = (CUlimit)Int_val(limit_val);
  size_t limit_value =
      (size_t)Int64_val(value_val);  // Renamed from 'value' to 'limit_value'
  CUresult res = cuCtxSetLimit(limit, limit_value);
  if (res != CUDA_SUCCESS) caml_failwith("cuCtxSetLimit failed");
  CAMLreturn(Val_unit);
}

value caml_cu_ctx_get_limit(value limit_val) {
  CAMLparam1(limit_val);
  CUlimit limit = (CUlimit)Int_val(limit_val);
  size_t limit_value;  // Renamed from 'value' to 'limit_value'
  CUresult res = cuCtxGetLimit(&limit_value, limit);
  if (res != CUDA_SUCCESS) caml_failwith("cuCtxGetLimit failed");
  CAMLreturn(caml_copy_int64((int64_t)limit_value));
}

/* Stream Management */
value caml_cu_stream_attach_mem_async(value stream_val, value dptr_val,
                                      value length_val, value flags_val) {
  CAMLparam4(stream_val, dptr_val, length_val, flags_val);
  CUstream stream = (CUstream)(uintptr_t)Int64_val(stream_val);
  CUdeviceptr dptr = (CUdeviceptr)Int64_val(dptr_val);
  size_t length = (size_t)Int64_val(length_val);
  unsigned int flags = Int_val(flags_val);
  CUresult res = cuStreamAttachMemAsync(stream, dptr, length, flags);
  if (res != CUDA_SUCCESS) caml_failwith("cuStreamAttachMemAsync failed");
  CAMLreturn(Val_unit);
}

value caml_cu_stream_create_with_priority(value flags_val, value priority_val) {
  CAMLparam2(flags_val, priority_val);
  unsigned int flags = Int_val(flags_val);
  int priority = Int_val(priority_val);
  CUstream stream;
  CUresult res = cuStreamCreateWithPriority(&stream, flags, priority);
  if (res != CUDA_SUCCESS) caml_failwith("cuStreamCreateWithPriority failed");
  CAMLreturn(caml_copy_int64((int64_t)(uintptr_t)stream));
}

value caml_cu_stream_destroy(value stream_val) {
  CAMLparam1(stream_val);
  CUstream stream = (CUstream)(uintptr_t)Int64_val(stream_val);
  CUresult res = cuStreamDestroy(stream);
  if (res != CUDA_SUCCESS) caml_failwith("cuStreamDestroy failed");
  CAMLreturn(Val_unit);
}

value caml_cu_stream_get_ctx(value stream_val) {
  CAMLparam1(stream_val);
  CUstream stream = (CUstream)(uintptr_t)Int64_val(stream_val);
  CUcontext ctx;
  CUresult res = cuStreamGetCtx(stream, &ctx);
  if (res != CUDA_SUCCESS) caml_failwith("cuStreamGetCtx failed");
  CAMLreturn(caml_copy_int64((int64_t)(uintptr_t)ctx));
}

value caml_cu_stream_get_id(value stream_val) {
  CAMLparam1(stream_val);
  CUstream stream = (CUstream)(uintptr_t)Int64_val(stream_val);
  unsigned long long id;
  CUresult res = cuStreamGetId(stream, &id);
  if (res != CUDA_SUCCESS) caml_failwith("cuStreamGetId failed");
  CAMLreturn(caml_copy_int64((int64_t)id));
}

value caml_cu_stream_query(value stream_val) {
  CAMLparam1(stream_val);
  CUstream stream = (CUstream)(uintptr_t)Int64_val(stream_val);
  CUresult res = cuStreamQuery(stream);
  CAMLreturn(Val_int(res == CUDA_SUCCESS ? 1 : 0));
}

value caml_cu_stream_synchronize(value stream_val) {
  CAMLparam1(stream_val);
  CUstream stream = (CUstream)(uintptr_t)Int64_val(stream_val);
  CUresult res = cuStreamSynchronize(stream);
  if (res != CUDA_SUCCESS) caml_failwith("cuStreamSynchronize failed");
  CAMLreturn(Val_unit);
}

/* Event Management */
value caml_cu_event_create(value flags_val) {
  CAMLparam1(flags_val);
  unsigned int flags = Int_val(flags_val);
  CUevent event;
  CUresult res = cuEventCreate(&event, flags);
  if (res != CUDA_SUCCESS) caml_failwith("cuEventCreate failed");
  CAMLreturn(caml_copy_int64((int64_t)(uintptr_t)event));
}

value caml_cu_event_destroy(value event_val) {
  CAMLparam1(event_val);
  CUevent event = (CUevent)(uintptr_t)Int64_val(event_val);
  CUresult res = cuEventDestroy(event);
  if (res != CUDA_SUCCESS) caml_failwith("cuEventDestroy failed");
  CAMLreturn(Val_unit);
}

value caml_cu_event_elapsed_time(value start_val, value end_val) {
  CAMLparam2(start_val, end_val);
  CUevent start = (CUevent)(uintptr_t)Int64_val(start_val);
  CUevent end = (CUevent)(uintptr_t)Int64_val(end_val);
  float ms;
  CUresult res = cuEventElapsedTime(&ms, start, end);
  if (res != CUDA_SUCCESS) caml_failwith("cuEventElapsedTime failed");
  CAMLreturn(caml_copy_double((double)ms));
}

value caml_cu_event_record_with_flags(value event_val, value stream_val,
                                      value flags_val) {
  CAMLparam3(event_val, stream_val, flags_val);
  CUevent event = (CUevent)(uintptr_t)Int64_val(event_val);
  CUstream stream = (CUstream)(uintptr_t)Int64_val(stream_val);
  unsigned int flags = Int_val(flags_val);
  CUresult res = cuEventRecordWithFlags(event, stream, flags);
  if (res != CUDA_SUCCESS) caml_failwith("cuEventRecordWithFlags failed");
  CAMLreturn(Val_unit);
}

value caml_cu_event_query(value event_val) {
  CAMLparam1(event_val);
  CUevent event = (CUevent)(uintptr_t)Int64_val(event_val);
  CUresult res = cuEventQuery(event);
  CAMLreturn(Val_int(res == CUDA_SUCCESS ? 1 : 0));
}

value caml_cu_event_synchronize(value event_val) {
  CAMLparam1(event_val);
  CUevent event = (CUevent)(uintptr_t)Int64_val(event_val);
  CUresult res = cuEventSynchronize(event);
  if (res != CUDA_SUCCESS) caml_failwith("cuEventSynchronize failed");
  CAMLreturn(Val_unit);
}

value caml_cu_stream_wait_event(value stream_val, value event_val,
                                value flags_val) {
  CAMLparam3(stream_val, event_val, flags_val);
  CUstream stream = (CUstream)(uintptr_t)Int64_val(stream_val);
  CUevent event = (CUevent)(uintptr_t)Int64_val(event_val);
  unsigned int flags = Int_val(flags_val);
  CUresult res = cuStreamWaitEvent(stream, event, flags);
  if (res != CUDA_SUCCESS) caml_failwith("cuStreamWaitEvent failed");
  CAMLreturn(Val_unit);
}