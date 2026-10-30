(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

external deeper_c : int -> (unit -> 'a) -> 'a = "nx_stack_support_deeper"

let deeper n f =
  if n < 0 || n > 4096 then invalid_arg "Nx_stack_support.deeper";
  deeper_c n f
