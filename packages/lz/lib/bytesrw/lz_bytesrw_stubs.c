/* Copies between bytesrw's slices, which are bytes, and the bigarrays lz's
   steps read and write. The OCaml side checks every range; a copy is at
   most a run or a staged chunk, short enough to keep the runtime lock. An
   empty copy returns before touching its pointers: an empty bigarray's data
   may be NULL, and C forbids even adding zero to NULL. */

#define CAML_NAME_SPACE
#include <caml/bigarray.h>
#include <caml/mlvalues.h>

#include <string.h>

/* [stage b first dst n] copies the [n] bytes of [b] from [first] to the
   start of [dst]. */
CAMLprim value lz_bytesrw_stage(value b, value first, value dst, value n) {
  if (Long_val(n) == 0) return Val_unit;
  memcpy(Caml_ba_data_val(dst), Bytes_val(b) + Long_val(first),
         (size_t)Long_val(n));
  return Val_unit;
}

/* [unstage src first b n] copies the [n] bytes of [src] from [first] to
   the start of [b]. */
CAMLprim value lz_bytesrw_unstage(value src, value first, value b, value n) {
  if (Long_val(n) == 0) return Val_unit;
  memcpy(Bytes_val(b), (const unsigned char *)Caml_ba_data_val(src) +
                           Long_val(first),
         (size_t)Long_val(n));
  return Val_unit;
}
