/* Stubs for staging.ml: the C blits between a bigarray and bytes that a
   bytesrw filter over lz would stage its slices with. Neither allocates or
   releases the lock: a [bytes] may move during a collection. */

#define CAML_NAME_SPACE
#include <caml/bigarray.h>
#include <caml/mlvalues.h>

#include <string.h>
#include <time.h>

value lzm_clock(value unit) {
  (void)unit;
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return Val_long((intnat)t.tv_sec * 1000000000 + t.tv_nsec);
}

/* [n] bytes of [ba] from [first] to [b] from [at]. */
value lzm_ba_to_bytes(value ba, value first, value b, value at, value n) {
  memcpy(Bytes_val(b) + Long_val(at),
         (const char *)Caml_ba_data_val(ba) + Long_val(first),
         (size_t)Long_val(n));
  return Val_unit;
}

/* [n] bytes of [b] from [at] to [ba] from [first]. */
value lzm_bytes_to_ba(value b, value at, value ba, value first, value n) {
  memcpy((char *)Caml_ba_data_val(ba) + Long_val(first),
         Bytes_val(b) + Long_val(at), (size_t)Long_val(n));
  return Val_unit;
}
