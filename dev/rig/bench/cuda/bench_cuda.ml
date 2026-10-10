(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* GPU 0 through the driver, its work submitted through rig, each row beside the
   CUDA calls that bound it, made from C on streams of their own: a release by a
   stream write and a spin on the word; a queue switch by an event wait, the
   write and an event record; foreign waits, which rig makes only across
   devices, through the driver's C submit, by one batch of memory operations;
   launches, copies, allocations and page-locking by the same calls. A row waits
   by spinning on the word. Each case opens its device in its own worker, so
   that no process forks after CUDA started. Without an NVIDIA GPU the suite has
   no rows. *)

module C = Rig_cuda
module S = Rig_cuda_support
module H = Rig_gpu_support.Host

external names : unit -> string array = "rig_cuda_bench_names"
external bind : nativeint array -> unit = "rig_cuda_bench_bind"
external start : unit -> unit = "rig_cuda_bench_start"
external floor_release : int -> unit = "rig_cuda_bench_release"
external floor_switch : unit -> unit = "rig_cuda_bench_switch"
external floor_waits : int -> int -> unit = "rig_cuda_bench_waits"
external floor_launch : nativeint -> int -> unit = "rig_cuda_bench_launch"
external floor_graph : nativeint -> int -> unit = "rig_cuda_bench_graph"

external floor_graph_launch : bool -> unit = "rig_cuda_bench_graph_launch"
external buffer : bool -> int -> nativeint = "rig_cuda_bench_buffer"

external floor_copy : nativeint -> nativeint -> int -> unit
  = "rig_cuda_bench_copy"

external floor_alloc : int -> unit = "rig_cuda_bench_alloc"
external floor_map_host : int -> int -> unit = "rig_cuda_bench_map_host"

external entry_waits : nativeint -> int -> int -> int -> unit
  = "rig_cuda_bench_entry_waits"

let fixtures = "../../test/cuda/fixtures"
let kib = 1024
let mib = 1024 * kib

(* A device and the run its row submits with. *)
type dev = { d : Rig.t; g : C.t; run : Rig.Submission.Run.t; mutable v : int }

let get = function Ok x -> x | Error why -> failwith why
let host r = Option.get (C.locate r).host

(* GPU 0, opened through rig. *)
let dev () =
  let { S.d; g } = S.open_ () in
  { d; g; run = Rig.Submission.Run.make (); v = 0 }

(* The prepared submission of [parts] on [t]. *)
let prepare t parts = Rig.Submission.make t.d parts

let submit t s =
  let p = Rig.submit s ~run:t.run ~buffers:[||] ~waits:[||] in
  t.v <- Rig.Point.value p

let wait t = Rig.wait t.d t.v

(* Waits for the value the driver's own entry was handed, which rig did not
   assign. *)
let spin t =
  while C.signaled t.g < t.v do
    Domain.cpu_relax ()
  done

let run t s =
  submit t s;
  wait t

(* The floor's calls, found by a device's capability. *)
let floor () =
  let g = get (C.open_ 0) in
  let symbol = (C.capability g).symbol in
  bind (Array.map (fun n -> Option.get (symbol n)) (names ()));
  start ();
  g

let row name setup f = Thumper.bench_with_setup ~setup name f
let empty g = snd (S.kernels ~dir:fixtures g) "empty"

(* [n] launches of [empty] over one thread, as parts of COMPUTE:0 loaded by rig,
   their blocks stored in [t]'s run. *)
let launches t n =
  let module Sub = Rig.Submission in
  let bin = S.fixture ~dir:fixtures "kernels.ptx" in
  let image = get (Rig.Image.load t.d bin) in
  let launch =
    Sub.Launch { image; kernel = "empty"; params = 16; refs = [||] }
  in
  let s =
    prepare t
      (Array.make n { Sub.queue = "COMPUTE:0"; after = [||]; work = launch })
  in
  for i = 0 to n - 1 do
    let b = Sub.block s i in
    Sub.Run.groups t.run b 1 1 1;
    Sub.Run.threads t.run b 1 1 1
  done;
  s

let release_rows =
  let empty () =
    let t = dev () in
    (t, prepare t [||])
  in
  let switching () =
    let t, empty = empty () in
    (t, prepare t [| S.part ~queue:"COPY:0" (S.failing 0) |], empty)
  in
  Thumper.group "release"
    [
      row "driver" empty (fun (t, s) -> run t s);
      row "floor" floor (fun _ -> floor_release 1);
      row "switch" switching (fun (t, copy, empty) ->
          run t (if t.v land 1 = 0 then copy else empty));
      row "floor-switch" floor (fun _ -> floor_switch ());
      row "no-wait-100" empty (fun (t, s) ->
          for _ = 1 to 100 do
            submit t s
          done;
          wait t);
      row "floor-no-wait-100" floor (fun _ -> floor_release 100);
    ]

let wait_rows =
  let word g =
    let w = Option.get (C.alloc g Pinned 8) in
    H.set64 (host w) 1;
    w
  in
  let waiting () =
    let t = dev () in
    (t, (C.facts t.g).edge, Option.get (C.locate (word t.g)).address)
  in
  let floor_waiting () = host (word (floor ())) in
  Thumper.group "waits"
    [
      row "4" waiting (fun (t, self, at) ->
          t.v <- t.v + 1;
          entry_waits self t.v at 4;
          spin t);
      row "floor-4" floor_waiting (fun at -> floor_waits at 4);
    ]

let kernels f n = Array.make n (S.kernel f 0 0)

(* A graph of [n] empty kernels, made through the capability, launched by a fill
   that updates no node or, [updated], every node's first argument to the run's
   parity: two prepared submissions, one per parity, alternate. *)
let graphing ?(updated = false) n () =
  let t = dev () in
  let f = empty t.g in
  let gr = get ((C.capability t.g).graph (kernels f n)) in
  let updates a =
    if updated then Array.init n (fun j -> (j, S.kernel f a 0)) else [||]
  in
  let prepared a =
    prepare t [| S.part ~queue:"COMPUTE:0" (S.graph_launch gr (updates a)) |]
  in
  (t, prepared 0, prepared 1)

let launch_rows =
  (* [n] launches as parts of one submission: [4096-parts] shows what the
     driver's ordering costs per part of a large submission. *)
  let launching n () =
    let t = dev () in
    (t, launches t n)
  in
  let floor_launching () = Nativeint.of_int (empty (floor ())) in
  let graph_launching (t, even, odd) =
    run t (if t.v land 1 = 0 then even else odd)
  in
  let floor_graphing n () =
    floor_graph (Nativeint.of_int (empty (floor ()))) n
  in
  Thumper.group "launch"
    [
      row "1" (launching 1) (fun (t, s) -> run t s);
      row "64" (launching 64) (fun (t, s) -> run t s);
      row "4096-parts" (launching 4096) (fun (t, s) -> run t s);
      row "submits-64" (launching 1) (fun (t, s) ->
          for _ = 1 to 64 do
            submit t s
          done;
          wait t);
      row "floor-1" floor_launching (fun f -> floor_launch f 1);
      row "floor-64" floor_launching (fun f -> floor_launch f 64);
      row "graph-1" (graphing 1) graph_launching;
      row "graph-64" (graphing 64) graph_launching;
      row "graph-64-updated" (graphing ~updated:true 64) graph_launching;
      row "floor-graph-1" (floor_graphing 1) (fun () ->
          floor_graph_launch false);
      row "floor-graph-64" (floor_graphing 64) (fun () ->
          floor_graph_launch false);
      row "floor-graph-64-updated" (floor_graphing 64) (fun () ->
          floor_graph_launch true);
    ]

(* The maker of a graph of 64 empty kernels and its release: link-time work. *)
let graph_rows =
  let making () =
    let t = dev () in
    ((C.capability t.g).graph, kernels (empty t.g) 64)
  in
  Thumper.group "graph"
    [ row "make-64" making (fun (graph, ks) -> (get (graph ks)).release ()) ]

let copy_rows =
  let n = 256 * mib in
  let copying (dst, src) () =
    let t = dev () in
    let create m = Rig.Buffer.create ~memory:m t.d n in
    let dst = create dst and src = create src in
    (t, prepare t [| S.copy ~queue:"COPY:0" ~dst src |])
  in
  let floor_copying (dst, src) () =
    ignore (floor ());
    (buffer dst n, buffer src n)
  in
  let copy name kinds host =
    [
      row name (copying kinds) (fun (t, s) -> run t s);
      row ("floor-" ^ name) (floor_copying host) (fun (dst, src) ->
          floor_copy dst src n);
    ]
  in
  Thumper.group "copy"
    (copy "h2d-256MiB" (Device, Pinned) (false, true)
    @ copy "d2h-256MiB" (Pinned, Device) (true, false)
    @ copy "d2d-256MiB" (Device, Device) (false, false))

let alloc_rows =
  let alloc name n =
    [
      row name dev (fun t -> C.free t.g (Option.get (C.alloc t.g Device n)));
      row ("floor-" ^ name) floor (fun _ -> floor_alloc n);
    ]
  in
  Thumper.group "alloc" (alloc "64KiB" (64 * kib) @ alloc "64MiB" (64 * mib))

let map_host_rows =
  let n = 256 * mib in
  Thumper.group "map-host"
    [
      row "256MiB"
        (fun () -> (dev (), H.pages n))
        (fun (t, p) -> C.free t.g (Option.get (C.map_host t.g p n)));
      row "floor-256MiB"
        (fun () -> (floor (), H.pages n))
        (fun (_, p) -> floor_map_host p n);
    ]

(* Loading an image of 140 kernels, the size of a kernel library's, every
   function's code placed (Rig_cuda.image), then unloading it. *)
let image_rows =
  let loading () =
    (dev (), In_channel.with_open_bin (Filename.concat fixtures "many.cubin")
       In_channel.input_all)
  in
  Thumper.group "image"
    [
      row "load-140" loading (fun (t, bin) ->
          match C.image t.g bin with
          | Ok (Loaded m) -> C.unload t.g m
          | Ok (Place _) -> failwith "a CUDA image is placed by CUDA"
          | Error why -> failwith why);
    ]

let () =
  S.hold ();
  if Sys.file_exists "/dev/nvidiactl" then
    exit
    @@ Thumper.run "rig_cuda"
         [
           release_rows;
           wait_rows;
           launch_rows;
           graph_rows;
           copy_rows;
           alloc_rows;
           map_host_rows;
           image_rows;
         ]
