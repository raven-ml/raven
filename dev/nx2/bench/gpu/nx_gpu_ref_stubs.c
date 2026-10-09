/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "ref.h"

static nx_ref_view view_of(value v) {
  nx_ref_view r;
  r.base = String_val(Field(v, 0));
  r.dtype = Int_val(Field(v, 1));
  for (int i = 0; i < 3; i++) r.s[i] = Long_val(Field(Field(v, 2), i));
  return r;
}

/* nx_ref_contract over the views [v_views] (a, b, y, then init if given),
   the extents [v_dims] (batch, m, n, k), and [v_l]: the accumulator,
   flush and the sample count. Allocates nothing before it reads. */
value nx_gpu_ref_contract(value v_views, value v_dims, value v_l) {
  CAMLparam3(v_views, v_dims, v_l);
  CAMLlocal1(r);
  nx_ref_view a = view_of(Field(v_views, 0)), b = view_of(Field(v_views, 1));
  nx_ref_view y = view_of(Field(v_views, 2)), in;
  const nx_ref_view *init = NULL;
  if (Wosize_val(v_views) > 3) in = view_of(Field(v_views, 3)), init = &in;
#define D(i) ((int64_t)Long_val(Field(v_dims, i)))
#define L(i) Long_val(Field(v_l, i))
  nx_ref_result x = nx_ref_contract(&a, &b, init, &y, D(0), D(1), D(2), D(3),
                                    (int)L(0), (int)L(1), L(2));
#undef D
#undef L
  r = caml_alloc_tuple(3);
  Store_field(r, 0, caml_copy_double(x.worst));
  Store_field(r, 1, Val_long(x.wrong));
  Store_field(r, 2, Val_long(x.at));
  CAMLreturn(r);
}
