/* Stubs for lock_cost.ml: a call that does nothing but release and
   reacquire the runtime lock, as stubs.c does around a kernel, beside the
   same call without the release. caml_release_runtime_system is a macro
   for caml_enter_blocking_section (OCaml 5.5's caml/threads.h), so one
   variant covers both. */

#define CAML_NAME_SPACE
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/signals.h>

#include <time.h>

value lzm_now(value unit) {
  (void)unit;
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return Val_long((intnat)t.tv_sec * 1000000000 + t.tv_nsec);
}

/* The call stubs.c makes below a kernel's threshold: its argument rooted,
   no release. */
value lzm_plain(value ba) {
  CAMLparam1(ba);
  CAMLreturn(Val_unit);
}

/* The call stubs.c makes above the threshold. */
value lzm_blocking(value ba) {
  CAMLparam1(ba);
  caml_enter_blocking_section();
  caml_leave_blocking_section();
  CAMLreturn(Val_unit);
}
