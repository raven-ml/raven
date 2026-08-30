(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

let round_up n align = (n + align - 1) / align * align

type t = { size : int; base : int; wrap : bool; mutable ptr : int }

let create ~size ?(base = 0) ?(wrap = true) () = { size; base; wrap; ptr = 0 }

let alloc t size ?(align = 1) () =
  if round_up t.ptr align + size > t.size then begin
    if not t.wrap then raise Out_of_memory;
    t.ptr <- 0
  end;
  let res = round_up t.ptr align in
  t.ptr <- res + size;
  res + t.base
