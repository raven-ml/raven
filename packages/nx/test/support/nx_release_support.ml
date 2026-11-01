(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

external install : unit -> unit = "nx_release_support_install" [@@noalloc]
external start : unit -> unit = "nx_release_support_start"
external stop : unit -> int = "nx_release_support_stop" [@@noalloc]
external releases : unit -> int = "nx_release_support_releases" [@@noalloc]
external longest : unit -> int = "nx_release_support_longest" [@@noalloc]
external stretches : unit -> int array = "nx_release_support_stretches"

external occupy : unit -> bool = "nx_release_support_occupy"
external occupied : unit -> bool = "nx_release_support_occupied" [@@noalloc]
external copy : unit -> int = "nx_release_support_copy"

let () = install ()

(* An efficiency core runs a held stretch up to about three times as long
   as the performance core the copy was timed on. *)
let loaded = 4
let bound () = loaded * copy ()

let watch f =
  start ();
  match f () with
  | () ->
      let cpu = stop () in
      (~releases:(releases ()), ~held:(longest ()), ~cpu)
  | exception e ->
      ignore (stop ());
      raise e

let held ~runs f =
  let run () =
    ignore (watch f);
    stretches ()
  in
  let least = run () in
  for _ = 2 to runs do
    let s = run () in
    if Array.length s <> Array.length least then
      failwith "Nx_release_support.held: runs of other stretches";
    Array.iteri (fun i d -> least.(i) <- min d least.(i)) s
  done;
  Array.fold_left max 0 least
