/*--------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC

  Byte spans of tensors: copies, byte swaps, Fortran-to-C reordering and
  writes to a file.
  --------------------------------------------------------------------------*/

#include "nx_io_codec.h"

#ifndef NX_IO_CODEC_NO_OCAML
#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>
#include <caml/unixsupport.h>
#endif

#include <errno.h>
#include <limits.h>
#include <string.h>
#include <unistd.h>

#ifndef NX_IO_CODEC_NO_OCAML
static void checked_span(value vbuf, value voff, value vlen,
                         const uint8_t **src, size_t *len) {
  intnat off_i = Long_val(voff);
  intnat len_i = Long_val(vlen);
  if (off_i < 0 || len_i < 0)
    caml_invalid_argument("Nx_io codec: negative byte span");
  size_t off = (size_t)off_i;
  size_t n = (size_t)len_i;
  size_t total = caml_ba_byte_size(Caml_ba_array_val(vbuf));
  if (off > total || n > total - off)
    caml_invalid_argument("Nx_io codec: byte span out of bounds");
  *src = (const uint8_t *)Caml_ba_data_val(vbuf) + off;
  *len = n;
}

CAMLprim value caml_nx_io_blit_bytes(value vsrc, value vsrc_off, value vdst,
                                     value vdst_off, value vlen) {
  CAMLparam5(vsrc, vsrc_off, vdst, vdst_off, vlen);
  const uint8_t *src;
  size_t len;
  checked_span(vsrc, vsrc_off, vlen, &src, &len);
  intnat dst_off_i = Long_val(vdst_off);
  if (dst_off_i < 0)
    caml_invalid_argument("Nx_io codec: negative destination offset");
  size_t dst_off = (size_t)dst_off_i;
  size_t dst_total = caml_ba_byte_size(Caml_ba_array_val(vdst));
  if (dst_off > dst_total || len > dst_total - dst_off)
    caml_invalid_argument("Nx_io codec: destination span out of bounds");
  uint8_t *dst = (uint8_t *)Caml_ba_data_val(vdst) + dst_off;
  caml_release_runtime_system();
  memmove(dst, src, len);
  caml_acquire_runtime_system();
  CAMLreturn(Val_unit);
}

CAMLprim value caml_nx_io_byteswap(value vbuf, value velement_size,
                                   value velements) {
  CAMLparam3(vbuf, velement_size, velements);
  intnat size_i = Long_val(velement_size);
  intnat elements_i = Long_val(velements);
  if (size_i <= 0 || elements_i < 0)
    caml_invalid_argument("Nx_io byteswap: invalid dimensions");
  size_t size = (size_t)size_i;
  size_t elements = (size_t)elements_i;
  size_t total = caml_ba_byte_size(Caml_ba_array_val(vbuf));
  if (elements != 0 && size > total / elements)
    caml_invalid_argument("Nx_io byteswap: span out of bounds");
  uint8_t *buf = (uint8_t *)Caml_ba_data_val(vbuf);
  caml_release_runtime_system();
  for (size_t element = 0; element < elements; element++) {
    uint8_t *p = buf + (element * size);
    for (size_t left = 0, right = size - 1; left < right; left++, right--) {
      uint8_t tmp = p[left];
      p[left] = p[right];
      p[right] = tmp;
    }
  }
  caml_acquire_runtime_system();
  CAMLreturn(Val_unit);
}

CAMLprim value caml_nx_io_reorder_fortran_to_c(value vsrc, value vsrc_off,
                                               value vdst, value vshape,
                                               value velement_size) {
  CAMLparam5(vsrc, vsrc_off, vdst, vshape, velement_size);
  intnat src_off_i = Long_val(vsrc_off);
  intnat size_i = Long_val(velement_size);
  if (src_off_i < 0 || size_i <= 0)
    caml_invalid_argument("Nx_io reorder: invalid byte span");
  size_t src_off = (size_t)src_off_i;
  size_t element_size = (size_t)size_i;
  mlsize_t rank = Wosize_val(vshape);
  if (rank > CAML_BA_MAX_NUM_DIMS)
    caml_invalid_argument("Nx_io reorder: rank exceeds Bigarray limit");
  size_t dims[CAML_BA_MAX_NUM_DIMS];
  size_t elements = 1;
  for (mlsize_t axis = 0; axis < rank; axis++) {
    intnat dim_i = Long_val(Field(vshape, axis));
    if (dim_i < 0)
      caml_invalid_argument("Nx_io reorder: negative dimension");
    size_t dim = (size_t)dim_i;
    if (dim != 0 && elements > SIZE_MAX / dim)
      caml_invalid_argument("Nx_io reorder: shape overflow");
    dims[axis] = dim;
    elements *= dim;
  }
  if (elements != 0 && element_size > SIZE_MAX / elements)
    caml_invalid_argument("Nx_io reorder: byte size overflow");
  size_t bytes = elements * element_size;
  size_t src_total = caml_ba_byte_size(Caml_ba_array_val(vsrc));
  size_t dst_total = caml_ba_byte_size(Caml_ba_array_val(vdst));
  if (src_off > src_total || bytes > src_total - src_off || bytes > dst_total)
    caml_invalid_argument("Nx_io reorder: byte span out of bounds");
  const uint8_t *src = (const uint8_t *)Caml_ba_data_val(vsrc) + src_off;
  uint8_t *dst = (uint8_t *)Caml_ba_data_val(vdst);

  caml_release_runtime_system();
  for (size_t c_index = 0; c_index < elements; c_index++) {
    size_t remaining = c_index;
    size_t coordinates[CAML_BA_MAX_NUM_DIMS];
    for (mlsize_t axis = rank; axis > 0; axis--) {
      size_t dim = dims[axis - 1];
      coordinates[axis - 1] = dim == 0 ? 0 : remaining % dim;
      if (dim != 0)
        remaining /= dim;
    }
    size_t f_index = 0;
    size_t f_stride = 1;
    for (mlsize_t axis = 0; axis < rank; axis++) {
      f_index += coordinates[axis] * f_stride;
      f_stride *= dims[axis];
    }
    memcpy(dst + (c_index * element_size), src + (f_index * element_size),
           element_size);
  }
  caml_acquire_runtime_system();
  CAMLreturn(Val_unit);
}

CAMLprim value caml_nx_io_write_all(value vfd, value vbuf, value voff,
                                    value vlen) {
  CAMLparam4(vfd, vbuf, voff, vlen);
  const uint8_t *src;
  size_t len;
  checked_span(vbuf, voff, vlen, &src, &len);
  nx_io_fd fd = Nx_io_fd_val(vfd);
  caml_release_runtime_system();
  int error = nx_io_write_all(fd, src, len);
  caml_acquire_runtime_system();
  if (error != 0)
    unix_error(error, "write", Nothing);
  CAMLreturn(Val_unit);
}

#endif
