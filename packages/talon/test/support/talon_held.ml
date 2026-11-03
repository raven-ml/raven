(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Talon
open Windtrap

let settled () =
  for _ = 1 to 4 do
    Gc.full_major ();
    ignore (Rig.Buffer.create Rig.host 0);
    Rig.free_cache Rig.host
  done;
  Rig.allocated Rig.host

(* The buffer and bytes of a host value's array. *)
let array (type v s) (x : (v, s, Nx.host) Nx.t) =
  match Nx.Repr.array x with
  | None -> []
  | Some a ->
      let n = Nx_array.Layout.numel (Nx_array.layout a) in
      [ (Nx_array.buffer a, Nx_array.Dtype.bytes (Nx_array.dtype a) n) ]

let rec arrays c =
  let validity v = Option.fold ~none:[] ~some:array v in
  match Column.layout c with
  | Fixed { validity = v; values = P x } -> validity v @ array x
  | Varsize { validity = v; offsets; child } ->
      validity v @ array offsets @ arrays child
  | Children { validity = v; fields; _ } ->
      validity v @ List.concat_map (fun (_, c) -> arrays c) fields

let made ?(inputs = []) c =
  let theirs = List.concat_map arrays inputs in
  let overlaps b (b', _) = Rig.Buffer.overlaps b b' in
  let count (seen, total) ((b, n) as a) =
    if List.exists (overlaps b) theirs || List.exists (overlaps b) seen then
      (seen, total)
    else (a :: seen, total + n)
  in
  snd (List.fold_left count ([], 0) (arrays c))

let check ?inputs make =
  ignore (make ());
  let before = settled () in
  let live, bytes =
    (fun () ->
      let c = make () in
      let live = settled () - before in
      let bytes = made ?inputs c in
      ignore (Sys.opaque_identity c);
      (live, bytes))
      ()
  in
  let after = settled () - before in
  (* What [make] reads, its inputs, stays live through the counts. *)
  ignore (Sys.opaque_identity (make, inputs));
  at_least ~msg:"bytes held while the column is live" int ~than:bytes live;
  equal ~msg:"bytes held once it is collected" int 0 after
