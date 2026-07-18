/*--------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
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

static void crc32_tables(uint32_t table[8][256]) {
  for (uint32_t i = 0; i < 256; i++) {
    uint32_t c = i;
    for (int bit = 0; bit < 8; bit++)
      c = (c >> 1) ^ (0xedb88320u & (0u - (c & 1u)));
    table[0][i] = c;
  }
  for (unsigned slice = 1; slice < 8; slice++)
    for (unsigned i = 0; i < 256; i++) {
      uint32_t previous = table[slice - 1][i];
      table[slice][i] = table[0][previous & 0xffu] ^ (previous >> 8);
    }
}

static uint32_t read_le32(const uint8_t *src) {
  return (uint32_t)src[0] | ((uint32_t)src[1] << 8) | ((uint32_t)src[2] << 16) |
         ((uint32_t)src[3] << 24);
}

static uint32_t crc32_update(const uint32_t table[8][256], uint32_t crc,
                             const uint8_t *src, size_t len) {
  while (len >= 8) {
    uint32_t first = crc ^ read_le32(src);
    uint32_t second = read_le32(src + 4);
    crc = table[7][first & 0xffu] ^ table[6][(first >> 8) & 0xffu] ^
          table[5][(first >> 16) & 0xffu] ^ table[4][first >> 24] ^
          table[3][second & 0xffu] ^ table[2][(second >> 8) & 0xffu] ^
          table[1][(second >> 16) & 0xffu] ^ table[0][second >> 24];
    src += 8;
    len -= 8;
  }
  while (len-- != 0)
    crc = table[0][(crc ^ *src++) & 0xffu] ^ (crc >> 8);
  return crc;
}

uint32_t nx_io_crc32(const uint8_t *src, size_t len) {
  uint32_t table[8][256];
  crc32_tables(table);
  uint32_t crc = crc32_update(table, 0xffffffffu, src, len);
  return crc ^ 0xffffffffu;
}

uint32_t nx_io_adler32(const uint8_t *src, size_t len) {
  const uint32_t modulus = 65521u;
  uint32_t a = 1u;
  uint32_t b = 0u;
  while (len != 0) {
    size_t chunk = len < 5552 ? len : 5552;
    len -= chunk;
    while (chunk-- != 0) {
      a += *src++;
      b += a;
    }
    a %= modulus;
    b %= modulus;
  }
  return (b << 16) | a;
}

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

CAMLprim value caml_nx_io_crc32(value vbuf, value voff, value vlen) {
  CAMLparam3(vbuf, voff, vlen);
  const uint8_t *src;
  size_t len;
  checked_span(vbuf, voff, vlen, &src, &len);
  caml_release_runtime_system();
  uint32_t crc = nx_io_crc32(src, len);
  caml_acquire_runtime_system();
  CAMLreturn(caml_copy_int32((int32_t)crc));
}

CAMLprim value caml_nx_io_adler32(value vbuf, value voff, value vlen) {
  CAMLparam3(vbuf, voff, vlen);
  const uint8_t *src;
  size_t len;
  checked_span(vbuf, voff, vlen, &src, &len);
  caml_release_runtime_system();
  uint32_t sum = nx_io_adler32(src, len);
  caml_acquire_runtime_system();
  CAMLreturn(caml_copy_int32((int32_t)sum));
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
  int fd = Int_val(vfd);
  int error = 0;
  size_t off = 0;
  caml_release_runtime_system();
  while (off < len) {
    ssize_t written = write(fd, src + off, len - off);
    if (written < 0 && errno == EINTR)
      continue;
    if (written <= 0) {
      error = written < 0 ? errno : EIO;
      break;
    }
    off += (size_t)written;
  }
  caml_acquire_runtime_system();
  if (error != 0)
    unix_error(error, "write", Nothing);
  CAMLreturn(Val_unit);
}

static int write_exact(int fd, const uint8_t *src, size_t len) {
  size_t off = 0;
  while (off < len) {
    ssize_t written = write(fd, src + off, len - off);
    if (written < 0 && errno == EINTR)
      continue;
    if (written <= 0)
      return written < 0 ? errno : EIO;
    off += (size_t)written;
  }
  return 0;
}

CAMLprim value caml_nx_io_store_to_fd(value vfd, value vprefix, value vbuf,
                                      value voff, value vlen) {
  CAMLparam5(vfd, vprefix, vbuf, voff, vlen);
  CAMLlocal2(vresult, vcrc);
  const uint8_t *src;
  size_t len;
  checked_span(vbuf, voff, vlen, &src, &len);
  size_t prefix_len = caml_string_length(vprefix);
  if (prefix_len > SIZE_MAX - len || prefix_len + len > (size_t)Max_long)
    caml_invalid_argument("Nx_io store: input is too large");
  uint8_t *prefix = prefix_len == 0 ? NULL : malloc(prefix_len);
  if (prefix_len != 0 && prefix == NULL)
    caml_raise_out_of_memory();
  if (prefix_len != 0)
    memcpy(prefix, String_val(vprefix), prefix_len);
  size_t total = prefix_len + len;
  int fd = Int_val(vfd);
  int error;
  uint32_t table[8][256];
  crc32_tables(table);
  caml_release_runtime_system();
  uint32_t crc = crc32_update(table, 0xffffffffu, prefix, prefix_len);
  crc = crc32_update(table, crc, src, len) ^ 0xffffffffu;
  error = write_exact(fd, prefix, prefix_len);
  if (error == 0)
    error = write_exact(fd, src, len);
  caml_acquire_runtime_system();
  free(prefix);
  if (error != 0)
    unix_error(error, "write", Nothing);
  vcrc = caml_copy_int32((int32_t)crc);
  vresult = caml_alloc_tuple(3);
  Store_field(vresult, 0, vcrc);
  Store_field(vresult, 1, Val_long(total));
  Store_field(vresult, 2, Val_long(total));
  CAMLreturn(vresult);
}
#endif
