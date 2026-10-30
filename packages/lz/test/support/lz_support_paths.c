/* The CRC-32 and Adler-32 paths this host can run (lib/checksum_paths.h),
   for test_checksum: Lz reaches only the fastest, and the tests hold each
   path, the generic one included, to the same references. */

#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "checksum_paths.h"

static size_t paths(value kind, const struct lz_checksum_path **p) {
  return Int_val(kind) == 0 ? lz_crc32_paths(p) : lz_adler32_paths(p);
}

/* The names of the paths of [kind]: 0 for CRC-32, 1 for Adler-32. */
CAMLprim value lz_support_paths(value kind) {
  CAMLparam1(kind);
  CAMLlocal2(names, name);
  const struct lz_checksum_path *p;
  size_t n = paths(kind, &p);
  names = caml_alloc(n, 0);
  for (size_t i = 0; i < n; i++) {
    name = caml_copy_string(p[i].name);
    Store_field(names, i, name);
  }
  CAMLreturn(names);
}

/* Path [i] of [kind] over all of [b], from the running value [c]. */
CAMLprim value lz_support_path(value kind, value i, value c, value b) {
  const struct lz_checksum_path *p;
  size_t n = paths(kind, &p);
  size_t k = (size_t)Long_val(i);
  if (k >= n) caml_invalid_argument("Lz_support: no such path");
  const uint8_t *data = Caml_ba_data_val(b);
  size_t len = (size_t)Caml_ba_array_val(b)->dim[0];
  static const uint8_t empty;
  uint32_t r = p[k].fn((uint32_t)Int32_val(c), len == 0 ? &empty : data, len);
  return caml_copy_int32((int32_t)r);
}
