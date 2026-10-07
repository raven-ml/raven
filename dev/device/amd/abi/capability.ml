(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type compute = Pm4 | Aql of { scratch : int -> (unit, string) result }

type t = {
  gpu : Gpu.t;
  clock_hz : int;
  compute : compute;
  place : nativeint;
  segment : nativeint;
}

let key = Type.Id.make ()
