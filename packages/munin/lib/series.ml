(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = {
  steps : int array;
  times : Float.Array.t;
  values : Float.Array.t;
  length : int;
}
