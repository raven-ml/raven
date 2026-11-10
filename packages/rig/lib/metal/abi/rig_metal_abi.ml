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

external reraise : exn -> 'a = "%reraise"

(* A finaliser runs [collect ()] with the thread's signals blocked, its
   waits included ([rig_shield] and [rig_unshield] in [rig_edge.h]), so that
   no signal's exception escapes it into the code it interrupted. *)
let collected collect _ =
  Rig_edge_primitive.shield ();
  match collect () with
  | () -> Rig_edge_primitive.unshield ()
  | exception e ->
      Rig_edge_primitive.unshield ();
      reraise e

let icb ~handle ~commands ~collect =
  let b = { handle; commands } in
  Gc.finalise (collected collect) b;
  b

type t = {
  align : int;
  icb : nativeint -> dispatch array -> (icb, string) result;
  split : nativeint;
}

let key : t Type.Id.t = Type.Id.make ()
let declare c = Rig_edge.Capability (key, c)
