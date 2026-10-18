/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Read claims around a kernel (kernels.ml).

   A kernel claims its operands' memory for reading while it runs, every
   operand or none, so that no compiled call on another domain donates it
   meanwhile. A kernel the host runs also waits for the work that wrote the
   memory; a device's kernel is ordered by its device. */

#include <stdio.h>

#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "rig.h"

/* kernels.ml's [follows], by the rank of its constructors. */
enum { FOLLOWS_WAITS, FOLLOWS_DEVICE_ORDERS };
#define Waits_val(v) (Int_val(v) == FOLLOWS_WAITS)

/* The operands claim in runs of this many buffers, gathered on the stack, so
   that no number of operands overflows it. */
#define RUN 16

/* The [k]th buffer of [bs]: a C array of roots, or a root of an OCaml array,
   read again after each wait, which may move it. */
typedef value (*nth)(const value *bs, int k);

static value nth_c(const value *bs, int k) { return bs[k]; }
static value nth_ocaml(const value *bs, int k) { return Field(*bs, k); }

static void release_first(int n, const value *bs, nth get) {
  for (int k = 0; k < n; k++) rig_buffer_release(get(bs, k));
}

/* Claims the memory of the [n] buffers of [bs] for reading, every claim or
   none; with [wait], also waits for their last writes. Raises
   Invalid_argument for a dead buffer or memory held exclusive, and Rig.Lost
   where the memory depends on a lost device, holding no claim then. Each
   run's waits come before the next run's claims. */
static void claim(int n, const value *bs, nth get, int wait) {
  value buffer[RUN];
  enum rig_access access[RUN];
  enum rig_claim answer[RUN];
  for (int k = 0; k < RUN; k++) access[k] = RIG_READ;
  for (int at = 0; at < n; at += RUN) {
    int m = n - at < RUN ? n - at : RUN;
    for (int j = 0; j < m; j++) buffer[j] = get(bs, at + j);
    int refused = rig_buffers_claim(m, buffer, access, answer);
    if (refused >= 0) {
      release_first(at, bs, get);
      value b = get(bs, at + refused);
      if (answer[refused] == RIG_DEAD) {
        char why[256];
        snprintf(why, sizeof why, "%s", rig_buffer_why(b));
        caml_invalid_argument(why);
      }
      caml_invalid_argument(
          "Nx: storage is held exclusive by a consuming call");
    }
    for (int j = 0; j < m; j++) {
      if (answer[j] == RIG_CLAIMED || (answer[j] == RIG_WAIT && !wait))
        continue;
      caml_result r = rig_buffer_wait(get(bs, at + j), RIG_READ);
      if (caml_result_is_exception(r)) {
        release_first(at + m, bs, get);
        caml_get_value_or_raise(r);
      }
    }
  }
}

value caml_nx_claim(value a, value follows) {
  CAMLparam2(a, follows);
  claim(1, &a, nth_c, Waits_val(follows));
  CAMLreturn(Val_unit);
}

value caml_nx_claim2(value a, value b, value follows) {
  CAMLparam3(a, b, follows);
  CAMLlocalN(bs, 2);
  bs[0] = a;
  bs[1] = b;
  claim(2, bs, nth_c, Waits_val(follows));
  CAMLreturn(Val_unit);
}

value caml_nx_claim3(value a, value b, value c, value follows) {
  CAMLparam4(a, b, c, follows);
  CAMLlocalN(bs, 3);
  bs[0] = a;
  bs[1] = b;
  bs[2] = c;
  claim(3, bs, nth_c, Waits_val(follows));
  CAMLreturn(Val_unit);
}

/* [v_bs] is an OCaml array, a root through which each buffer is read. */
value caml_nx_claim_all(value v_bs, value follows) {
  CAMLparam2(v_bs, follows);
  claim((int)Wosize_val(v_bs), &v_bs, nth_ocaml, Waits_val(follows));
  CAMLreturn(Val_unit);
}

value caml_nx_release(value a) {
  rig_buffer_release(a);
  return Val_unit;
}
