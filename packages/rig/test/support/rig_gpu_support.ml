(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap

(* Host memory *)

module Host = struct
  external at : Rig.Buffer.t -> int = "rig_gpu_test_buffer_at" [@@noalloc]

  external data : ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t -> int
    = "rig_gpu_host_data"
  [@@noalloc]

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
  let word () = Nativeint.of_int (pages 8)
end

let opened (o : _ Rig_edge.opener) = o.make ~word:(Host.word ())

let still ?msg w x f ~ms =
  let t0 = Rig.Profile.now () in
  while Rig.Profile.now () - t0 < ms * 1_000_000 do
    equal ?msg w x (f ())
  done

(* A GPU *)

module type Gpu = sig
  module D : Rig_edge.Driver

  val class_ : string
  val present : unit -> bool
  val open_ : word:nativeint -> (D.t, string) result
end

(* Work *)

type kept = Kept : 'a -> kept

type work = {
  part : int -> Rig.Submission.part;
  buffers : Rig.Buffer.t array;
  fixed : (Rig.Buffer.t * Rig.access) list;
  keep : kept;
  init : Rig.Submission.t -> Rig.Submission.Run.t -> int -> unit;
}

let work part =
  {
    part = (fun _ -> part);
    buffers = [||];
    fixed = [];
    keep = Kept ();
    init = (fun _ _ _ -> ());
  }

let prepare ?(fixed = []) d ws =
  let module Sub = Rig.Submission in
  let base = ref 0 in
  let part w =
    let p = w.part !base in
    base := !base + Array.length w.buffers;
    p
  in
  let parts = Array.of_list (List.map part ws) in
  let buffers = Array.concat (List.map (fun w -> w.buffers) ws) in
  let fixed = fixed @ List.concat_map (fun w -> w.fixed) ws in
  let s = Sub.make ~keep:ws ~fixed d parts in
  let run = Sub.Run.make () in
  List.iteri (fun i w -> w.init s run i) ws;
  (s, run, buffers)

(* [kernel] of [image] over [groups] of [threads], each of [uses] a buffer
   referenced at a parameter byte with an access, and [params] bytes of
   parameters [store] stores. *)
let launching image kernel ~params ?(groups = 1) ?(threads = 1) uses store =
  let module Sub = Rig.Submission in
  let part base =
    let refs =
      Array.mapi
        (fun i (_, (a : Rig.access), at) ->
          match a with
          | Read -> Sub.read (base + i) ~at
          | Read_write -> Sub.write (base + i) ~at)
        uses
    in
    Sub.launch ~refs image kernel ~params
  in
  let init s run i =
    let b = Sub.block s i in
    Sub.Run.groups run b groups 1 1;
    Sub.Run.threads run b threads 1 1;
    store run b
  in
  {
    part;
    buffers = Array.map (fun (b, _, _) -> b) uses;
    fixed = [];
    keep = Kept ();
    init;
  }

(* Parameters from a C fixture, padded to 32-bit words. *)
let params run block bytes =
  let n = String.length bytes in
  let padding = (4 - (n mod 4)) mod 4 in
  let bytes =
    if padding = 0 then bytes else bytes ^ String.make padding '\000'
  in
  for i = 0 to (String.length bytes / 4) - 1 do
    Rig.Submission.Run.int32 run block (4 * i)
      (Int32.to_int (String.get_int32_ne bytes (4 * i)))
  done

module type S = sig
  type gpu
  type t = { d : Rig.t; g : gpu }

  val class_ : string
  val present : unit -> bool
  val open_ : unit -> t
  val open_as : (module Rig_edge.Driver with type t = gpu) -> t
  val open_own : (module Rig_edge.Driver with type t = gpu) -> t
  val close : t -> unit
  val with_ : (t -> 'a) -> 'a
  val driver : unit -> gpu
  val stop_driver : gpu -> unit
  val handed : string -> t
  val with_driver : (gpu -> 'a) -> 'a
  val release : unit -> unit
  val submit : t -> Rig.Submission.part array -> Rig.Point.t
  val submit_work : t -> work list -> Rig.Point.t
end

module Make (G : Gpu) = struct
  type gpu = G.D.t
  type t = { d : Rig.t; g : gpu }

  let class_ = G.class_
  let name = G.class_ ^ ":test"
  let present = G.present

  (* What the last open made, until it is ended: the next open ends one a failed
     test left, as a GPU that has one device at a time needs. *)
  type opened = Nothing | In_rig of Rig.t | Alone of gpu

  let opened = ref Nothing

  let release () =
    (match !opened with
    | In_rig d -> Rig.close d
    | Alone g -> G.D.stop g ~fault:None
    | Nothing -> ());
    opened := Nothing

  (* The driver's device, once the last open's is ended. *)
  let prepare_open () =
    if not (present ()) then
      skip
        ~reason:(Printf.sprintf "the machine has no %s GPU rig drives" G.class_)
        ();
    release ()

  let open_driver () =
    prepare_open ();
    match G.open_ ~word:(Host.word ()) with
    | Ok g -> g
    | Error why -> failf "%s: %s" name why

  let open_as driver =
    prepare_open ();
    let raw = ref None in
    let make ~word =
      Result.map
        (fun g ->
          raw := Some g;
          g)
        (G.open_ ~word)
    in
    match Rig.open_ { name; driver; make } with
    | Ok d ->
        opened := In_rig d;
        { d; g = Option.get !raw }
    | Error why -> failf "%s in rig: %s" name why

  let open_ () = open_as (module G.D)

  (* One word outside rig's array serves every open: each [make] copies the
     slot word's value into it first, and a device of a word of its own starts
     its own from that value. *)
  let outside = lazy (Nativeint.to_int (Host.word ()))

  let open_own driver =
    prepare_open ();
    let raw = ref None in
    let make ~word =
      let at = Lazy.force outside in
      Host.set64 at (Host.get64 (Nativeint.to_int word));
      Result.map
        (fun g ->
          raw := Some g;
          g)
        (G.open_ ~word:(Nativeint.of_int at))
    in
    match Rig.open_ { name; driver; make } with
    | Ok d ->
        opened := In_rig d;
        { d; g = Option.get !raw }
    | Error why -> failf "%s in rig on a word of its own: %s" name why

  let close t =
    Rig.close t.d;
    match !opened with In_rig d when d == t.d -> opened := Nothing | _ -> ()

  let with_ f =
    let t = open_ () in
    Fun.protect ~finally:(fun () -> close t) (fun () -> f t)

  let driver () =
    let g = open_driver () in
    opened := Alone g;
    g

  let stop_driver g =
    (match !opened with Alone a when a == g -> opened := Nothing | _ -> ());
    G.D.stop g ~fault:None

  let handed name =
    prepare_open ();
    let raw = ref None in
    let make ~word =
      Result.map
        (fun g ->
          raw := Some g;
          g)
        (G.open_ ~word)
    in
    match Rig.open_ { name; driver = (module G.D); make } with
    | Ok d ->
        let g = Option.get !raw in
        opened := Alone g;
        { d; g }
    | Error why -> failf "%s in rig as %s: %s" name name why

  let with_driver f =
    let g = driver () in
    Fun.protect ~finally:(fun () -> stop_driver g) (fun () -> f g)

  let submit t ps =
    let s = Rig.Submission.make t.d ps in
    let run = Rig.Submission.Run.make () in
    Rig.submit s ~run ~buffers:[||]

  let submit_work t ws =
    let s, run, buffers = prepare t.d ws in
    Rig.submit s ~run ~buffers
end

(* Watching a driver's word *)

module Watch (D : Rig_edge.Driver) = struct
  include D

  (* Each device's word, as its facts give it, with the frees of it and the
     reads that followed one, under [lock]. *)
  type watched = { h : D.t; word : D.region; frees : int; late_reads : int }

  let lock = Mutex.create ()
  let watched = ref []

  let update h f =
    Mutex.protect lock @@ fun () ->
    watched := List.map (fun w -> if w.h == h then f w else w) !watched

  let facts h =
    let f = D.facts h in
    Mutex.protect lock (fun () ->
        if not (List.exists (fun w -> w.h == h) !watched) then
          let w = { h; word = f.word; frees = 0; late_reads = 0 } in
          watched := w :: !watched);
    f

  let find h =
    Mutex.protect lock @@ fun () ->
    match List.find_opt (fun w -> w.h == h) !watched with
    | Some w -> w
    | None -> invalid_arg "Watch: a device whose facts were not read"

  let free h r =
    update h (fun w ->
        if w.word == r then { w with frees = w.frees + 1 } else w);
    D.free h r

  let signaled h =
    if (find h).frees > 0 then
      update h (fun w -> { w with late_reads = w.late_reads + 1 });
    D.signaled h

  let word_frees h = (find h).frees
  let late_reads h = (find h).late_reads
end

(* Conformance *)

module type Conformance = sig
  module D : Rig_edge.Driver
  include S with type gpu = D.t

  val binary : unit -> string * string list
  val second : word:nativeint -> (D.t, string) result option
  val copy_words : t -> dst:Rig.Buffer.t -> src:Rig.Buffer.t -> work
  val spin : t -> ns:int -> work
  val clock_error : since:int -> int
  val launch_binary : unit -> string option
  val wide_binary : t -> string option
end

let loader bin =
  let lock = Mutex.create () and loaded = ref [] in
  fun d ->
    Mutex.protect lock @@ fun () ->
    loaded := List.filter (fun (d, _) -> Rig.lost d = None) !loaded;
    match List.find_opt (fun (d', _) -> Rig.equal d d') !loaded with
    | Some (_, i) -> i
    | None -> (
        match Rig.Image.load d (bin ()) with
        | Ok i ->
            loaded := (d, i) :: !loaded;
            i
        | Error why -> failf "%a: %s" Rig.pp d why)

let arguments d s =
  let b = Rig.Buffer.create ~memory:Pinned d (String.length s) in
  Rig.Buffer.copy ~src:(Rig.Buffer.of_string s) ~dst:b;
  b
