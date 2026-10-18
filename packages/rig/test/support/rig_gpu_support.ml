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
  let run = Sub.Run.make s in
  List.iteri (fun i w -> w.init s run i) ws;
  (run, buffers)

(* [kernel] of [image] over [groups] of [threads], each of [uses] a buffer
   referenced at a parameter byte with an access, and [params] bytes of
   parameters [store] stores. *)
let launching image kernel ~params ?(groups = 1) ?(threads = 1) uses store =
  let module Sub = Rig.Submission in
  let fn =
    match Rig.Image.resolve image kernel with
    | Some fn -> fn
    | None -> failf "no function %s" kernel
  in
  let part base =
    let refs =
      Array.mapi
        (fun i (_, (a : Rig.access), at) ->
          match a with
          | Read -> Sub.read (base + i) ~at
          | Read_write -> Sub.write (base + i) ~at)
        uses
    in
    Sub.launch ~refs fn ~params
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
    let run = Rig.Submission.Run.make s in
    Rig.submit run ~buffers:[||]

  let submit_work t ws =
    let run, buffers = prepare t.d ws in
    Rig.submit run ~buffers
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

(* A driver's copies *)

external edge_copy : nativeint -> int -> int -> nativeint array -> int
  = "rig_gpu_edge_copy"

type raw = {
  dst : nativeint;
  dst_offset : int;
  src : nativeint;
  src_offset : int;
  width : int;
  rows : int;
  src_pitch : int;
  dst_pitch : int;
  local : [ `None | `Src | `Dst ];
  device_memory : bool;
  fill : char option;
}

type answer = Handed | Room of int | Failed

(* rig_edge.h's RIG_LOCAL_NONE, RIG_LOCAL_SRC and RIG_LOCAL_DST, and its room
   answers. *)
let hand edge ~v ~queue p =
  let n = Nativeint.of_int in
  let fields =
    [|
      p.dst;
      n p.dst_offset;
      p.src;
      n p.src_offset;
      n p.width;
      n p.rows;
      n p.src_pitch;
      n p.dst_pitch;
      (match p.local with `None -> 0n | `Src -> 1n | `Dst -> 2n);
      (if p.device_memory then 1n else 0n);
      (if p.fill = None then 0n else 1n);
      n (match p.fill with Some c -> Char.code c | None -> 0);
    |]
  in
  match edge_copy edge v queue fields with
  | 0 -> Handed
  | -1 -> Failed
  | r -> Room r

module Edge (D : Rig_edge.Driver) = struct
  type t = { g : D.t; mutable last : int; mutable regions : D.region list }
  type region = { r : D.region; memory : Rig_edge.memory; bytes : int }
  type side = Region of region | Host of int

  type part = {
    dst : side;
    dst_offset : int;
    src : side option;
    src_offset : int;
    width : int;
    rows : int;
    src_pitch : int;
    dst_pitch : int;
    fill : char option;
  }

  let facts t = D.facts t.g

  (* The value the device's word holds: the word's own bytes where the host
     addresses them, and the driver's read otherwise. *)
  let signaled t =
    match (D.locate (facts t).word).host with
    | Some at -> Host.get64 at
    | None -> D.signaled t.g

  let make g =
    let t = { g; last = 0; regions = [] } in
    t.last <- signaled t;
    t

  let alloc t memory bytes =
    match D.alloc t.g memory bytes with
    | Some r ->
        t.regions <- r :: t.regions;
        { r; memory; bytes }
    | None -> failf "no %d bytes of memory to allocate" bytes

  let release t =
    List.iter (D.free t.g) t.regions;
    t.regions <- []

  let handle = function
    | Region r -> (D.locate r.r).handle
    | Host at -> Nativeint.of_int at

  (* Waits for the word to reach [v], at most 10 s. *)
  let wait t v =
    let t0 = Rig.Profile.now () in
    while signaled t < v do
      if Rig.Profile.now () - t0 > 10_000_000_000 then
        failf "value %d within 10 s" v;
      D.sleep t.g ~seen:(signaled t) ~still_ms:100
    done

  let run t (p : part) =
    let queue =
      match (facts t).copy_engine with
      | Some q -> q
      | None -> invalid_arg "Edge.run: the device has no copy engine"
    in
    let device = function
      | Region r -> r.memory = Device
      | Host _ -> false
    in
    let local =
      match (p.src, p.dst) with
      | Some (Host _), _ -> `Src
      | _, Host _ -> `Dst
      | _ -> `None
    in
    let raw =
      {
        dst = handle p.dst;
        dst_offset = p.dst_offset;
        src = (match p.src with Some s -> handle s | None -> 0n);
        src_offset = p.src_offset;
        width = p.width;
        rows = p.rows;
        src_pitch = p.src_pitch;
        dst_pitch = p.dst_pitch;
        local;
        device_memory =
          device p.dst
          && (match p.src with Some s -> device s | None -> true);
        fill = p.fill;
      }
    in
    let v = t.last + 1 in
    let rec go () =
      match hand (facts t).edge ~v ~queue raw with
      | Handed -> ()
      | Room 1 ->
          D.sleep t.g ~seen:(signaled t) ~still_ms:10;
          go ()
      | Room 3 -> (
          match D.grow t.g with
          | Grown -> go ()
          | Short n -> failf "no %d bytes for the queues to grow" n)
      | Room _ -> failf "the device never runs the part"
      | Failed -> failf "the device failed the part"
    in
    go ();
    t.last <- v;
    wait t v

  let one_row ~dst ~dst_offset ~src ~src_offset n =
    {
      dst;
      dst_offset;
      src = Some src;
      src_offset;
      width = n;
      rows = 1;
      src_pitch = n;
      dst_pitch = n;
      fill = None;
    }

  (* [n] bytes of [r] from [at], through a copy into [Pinned] memory where
     the host does not address [r]. *)
  let rec read t r ~at n =
    match (D.locate r.r).host with
    | Some h -> Host.read (h + at) n
    | None ->
        let p = alloc t Pinned (Int.max n 1) in
        if n > 0 then
          run t
            (one_row ~dst:(Region p) ~dst_offset:0 ~src:(Region r)
               ~src_offset:at n);
        read t p ~at:0 n

  let rec write t r ~at s =
    let n = String.length s in
    match (D.locate r.r).host with
    | Some h -> Host.write (h + at) s
    | None ->
        let p = alloc t Pinned (Int.max n 1) in
        write t p ~at:0 s;
        if n > 0 then
          run t
            (one_row ~dst:(Region r) ~dst_offset:at ~src:(Region p)
               ~src_offset:0 n)
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
