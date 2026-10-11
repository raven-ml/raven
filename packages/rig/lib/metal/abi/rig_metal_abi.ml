(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type dispatch = {
  pipeline : int;
  offset : int;
  groups : int * int * int;
  threads : int * int * int;
}

type icb = { handle : nativeint; commands : nativeint array }

let icb ~handle ~commands ~collect =
  let b = { handle; commands } in
  Gc.finalise (fun _ -> collect ()) b;
  b

type t = {
  align : int;
  icb : nativeint -> dispatch array -> (icb, string) result;
  split : nativeint;
}

let key : t Type.Id.t = Type.Id.make ()
let declare c = Rig_edge.Capability (key, c)
