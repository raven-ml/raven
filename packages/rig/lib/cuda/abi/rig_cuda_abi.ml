(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type kernel = {
  func : int;
  grid : int * int * int;
  block : int * int * int;
  shared : int;
  args : string;
}

type graph = { handle : nativeint; nodes : nativeint array }

(* A finaliser runs [collect ()] with the thread's signals blocked, its
   waits included ([rig_shield] and [rig_unshield] in [rig_edge.h]), so that
   no signal's exception escapes it into the code it interrupted. *)
external shield : unit -> unit = "caml_rig_edge_shield" [@@noalloc]
external unshield : unit -> unit = "caml_rig_edge_unshield"
external reraise : exn -> 'a = "%reraise"

let collected collect _ =
  shield ();
  match collect () with
  | () -> unshield ()
  | exception e ->
      unshield ();
      reraise e

let graph ~handle ~nodes ~collect =
  let g = { handle; nodes } in
  Gc.finalise (collected collect) g;
  g

type t = {
  symbol : string -> nativeint option;
  graph : kernel array -> (graph, string) result;
}

let key : t Type.Id.t = Type.Id.make ()
let declare c = Rig_edge.Capability (key, c)
