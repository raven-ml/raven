(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

(* Host memory *)

module Host = struct
  external page_size : unit -> int = "rig_gpu_host_page" [@@noalloc]
  external pages : int -> bool -> int = "rig_gpu_host_pages"

  external free_pages : int -> int -> unit = "rig_gpu_host_free_pages"
  [@@noalloc]

  external get8 : int -> int = "rig_gpu_host_get8" [@@noalloc]
  external set8 : int -> int -> unit = "rig_gpu_host_set8" [@@noalloc]
  external get32 : int -> int = "rig_gpu_host_get32" [@@noalloc]
  external set32 : int -> int -> unit = "rig_gpu_host_set32" [@@noalloc]
  external get64 : int -> int = "rig_gpu_host_get64" [@@noalloc]
  external set64 : int -> int -> unit = "rig_gpu_host_set64" [@@noalloc]
  external read : int -> int -> string = "rig_gpu_host_read"
  external write : int -> string -> unit = "rig_gpu_host_write" [@@noalloc]

  let page = page_size ()
  let pages ?(read_only = false) n = pages n read_only
end

let still ?msg w x f ~ms =
  let t0 = Rig.Profile.now () in
  while Rig.Profile.now () - t0 < ms * 1_000_000 do
    equal ?msg w x (f ())
  done

(* A GPU *)

module type Gpu = sig
  module D : Rig.Driver

  val class_ : string
  val present : unit -> bool
  val open_ : unit -> (D.t, string) result
end

module type S = sig
  type gpu
  type t = { d : Rig.t; g : gpu }

  val hold : unit -> unit
  val open_ : unit -> t
  val close : t -> unit
  val release : unit -> unit
  val with_ : (t -> 'a) -> 'a
  val submit : t -> Rig.Submission.part array -> int
  val wait : t -> int -> unit
end

module Make (G : Gpu) = struct
  type gpu = G.D.t
  type t = { d : Rig.t; g : gpu }

  let name = G.class_ ^ ":test"
  let hold () = if G.present () then Rig_gpu_lock.hold ()

  (* The device the last open made, until it is closed: the next open closes
     one a failed test left. *)
  let opened = ref None

  let close t =
    Rig.close t.d;
    match !opened with Some d when d == t.d -> opened := None | _ -> ()

  let release () =
    Option.iter Rig.close !opened;
    opened := None

  let open_ () =
    if not (G.present ()) then
      skip ~reason:(Printf.sprintf "the machine has no %s GPU" G.class_) ();
    Rig_gpu_lock.hold ();
    release ();
    let g =
      match G.open_ () with Ok g -> g | Error why -> failf "%s: %s" name why
    in
    match Rig.open_ (module G.D) ~name (fun () -> Ok g) with
    | Ok d ->
        opened := Some d;
        { d; g }
    | Error why ->
        G.D.stop g;
        failf "%s in rig: %s" name why

  let with_ f =
    let t = open_ () in
    Fun.protect ~finally:(fun () -> close t) (fun () -> f t)

  let submit t ps =
    let s = Rig.Submission.make ~reads:0 ~writes:0 t.d ps in
    Rig.Point.value (Rig.submit s ~reads:[||] ~writes:[||] ~waits:[||])

  let wait t v = Rig.wait t.d v
end
