(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type compute = Pm4 | Aql of { scratch : int -> (unit, string) result }

type trace = {
  buffers : int;
  buffers_host : int;
  window : int;
  slots : int;
  engines : int;
  ends : int;
  ends_host : int;
}

type t = {
  gpu : Gpu.t;
  clock_hz : int;
  compute : compute;
  wgps : int array array;
  trace : unit -> (trace, string) result;
}

let key : t Type.Id.t = Type.Id.make ()
let declare c = Rig_edge.Capability (key, c)
